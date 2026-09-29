"""
orchestrator.py

Wires the whole flow together for one Jira ticket:

  Jira ticket
    -> Triage (code issue? yes/no)
        -> if NOT code issue -> Explanation -> post comment -> END
        -> if code issue:
            -> RAG retrieval (find relevant code)
            -> Git history (recent commits on suspect files)
            -> RCA (root cause hypothesis)
            -> if confidence high enough -> Fix proposal -> branch + draft MR
            -> else -> post RCA-only comment (no fix attempted), flag for human

This is a straightforward Python state machine rather than a full
LangGraph graph, so it's easy to read/debug in a local environment.
Swap in LangGraph's StateGraph later if you want visual graph tooling,
retries, or human-in-the-loop interrupts -- the agent functions themselves
don't need to change.
"""

from __future__ import annotations
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import yaml

from llm.provider import build_llm_provider
from integrations.jira_client import JiraClient
from integrations.gitlab_client import GitLabClient
from rag.indexer import CodebaseIndexer
from rag.retriever import CodebaseRetriever

from agents.triage_agent import run_triage
from agents.rca_agent import run_rca
from agents.fix_agent import run_fix_proposal, read_current_contents
from agents.explain_agent import run_explanation


@dataclass
class RunResult:
    ticket_key: str
    outcome: str          # "not_code_issue" | "fix_proposed" | "rca_only" | "error"
    details: dict


class L3Orchestrator:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        self.llm = build_llm_provider(self.cfg["llm"])

        self.jira = JiraClient(
            server=self.cfg["jira"]["server"],
            email=self.cfg["jira"]["email"],
            api_token=_resolve_env(self.cfg["jira"]["api_token"]),
        )

        self.gitlab = GitLabClient(
            url=self.cfg["gitlab"]["url"],
            private_token=_resolve_env(self.cfg["gitlab"]["private_token"]),
            clone_dir=self.cfg["gitlab"]["clone_dir"],
            default_branch=self.cfg["gitlab"]["default_branch"],
        )

        self.indexer = CodebaseIndexer(
            vector_db_path=self.cfg["rag"]["vector_db_path"],
            collection_name=self.cfg["rag"]["collection_name"],
            embedding_model=self.cfg["rag"]["embedding_model"],
            chunk_lines=self.cfg["rag"]["chunk_lines"],
            chunk_overlap_lines=self.cfg["rag"]["chunk_overlap_lines"],
            include_extensions=self.cfg["rag"]["include_extensions"],
            exclude_dirs=self.cfg["rag"]["exclude_dirs"],
        )
        self.retriever = CodebaseRetriever(
            vector_db_path=self.cfg["rag"]["vector_db_path"],
            collection_name=self.cfg["rag"]["collection_name"],
            embedding_model=self.cfg["rag"]["embedding_model"],
        )

        self.audit_log_path = Path(self.cfg["logging"]["audit_log_path"])
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)

    def index_project(self, jira_project_key: str) -> int:
        """Run this once (and periodically) to keep the RAG index fresh."""
        repo_path_str = self.cfg["gitlab"]["project_map"][jira_project_key]
        local_path = self.gitlab.ensure_local_clone(repo_path_str)
        count = self.indexer.index_repo(local_path, repo_label=repo_path_str)
        self._audit("index", {"project": jira_project_key, "chunks_indexed": count})
        return count

    def handle_ticket(self, ticket_key: str, jira_project_key: str) -> RunResult:
        ticket = self.jira.fetch_ticket(ticket_key)
        self._audit("ticket_fetched", {"ticket": ticket_key, "summary": ticket.summary})

        # 1. Triage
        triage = run_triage(self.llm, ticket)
        self._audit("triage", {"ticket": ticket_key, **asdict(triage)})

        if not triage.is_code_issue:
            explanation = run_explanation(self.llm, ticket, triage)
            self.jira.post_comment(
                ticket_key,
                f"[L3 Agent] Triage: {triage.issue_type}\n\n{explanation}",
            )
            self._audit("explanation_posted", {"ticket": ticket_key})
            return RunResult(ticket_key, "not_code_issue", {"triage": asdict(triage), "explanation": explanation})

        # 2. Retrieve relevant code
        repo_path_str = self.cfg["gitlab"]["project_map"][jira_project_key]
        local_repo_path = self.gitlab.ensure_local_clone(repo_path_str)
        chunks = self.retriever.search(
            query=triage.search_query,
            repo_label=repo_path_str,
            top_k=self.cfg["rag"]["top_k"],
        )
        self._audit("retrieval", {"ticket": ticket_key, "chunks_found": len(chunks)})

        # 3. Git history for suspect files
        suspect_files = sorted({c.file_path for c in chunks})
        commit_history = {
            f: self.gitlab.get_recent_commits(local_repo_path, f) for f in suspect_files
        }

        # 4. Root cause analysis
        rca = run_rca(self.llm, ticket, chunks, commit_history)
        self._audit("rca", {"ticket": ticket_key, **asdict(rca)})

        min_conf = self.cfg["workflow"]["min_confidence_to_propose_fix"]
        if rca.confidence < min_conf:
            self.jira.post_comment(
                ticket_key,
                f"[L3 Agent] Root Cause Analysis (confidence {rca.confidence:.0%} -- "
                f"below threshold for auto-fix, needs human investigation)\n\n"
                f"Root cause: {rca.root_cause}\n\nEvidence: {rca.evidence}",
            )
            self._audit("rca_only_posted", {"ticket": ticket_key})
            return RunResult(ticket_key, "rca_only", {"rca": asdict(rca)})

        # 5. Fix proposal
        current_contents = read_current_contents(local_repo_path, rca.suspect_files)
        fix = run_fix_proposal(self.llm, rca, current_contents)
        self._audit("fix_proposed", {"ticket": ticket_key, "files": list(fix.file_changes.keys())})

        if not fix.file_changes:
            self.jira.post_comment(
                ticket_key,
                f"[L3 Agent] Root cause identified but agent was not confident "
                f"enough to draft a safe fix automatically.\n\n"
                f"Root cause: {rca.root_cause}\n\nReasoning: {fix.explanation}",
            )
            return RunResult(ticket_key, "rca_only", {"rca": asdict(rca), "fix_explanation": fix.explanation})

        # 6. Branch + draft MR (human review required before merge)
        branch_name = f"{self.cfg['gitlab']['fix_branch_prefix']}/{ticket_key.lower()}-{int(time.time())}"
        self.gitlab.create_fix_branch_and_commit(
            project_path=repo_path_str,
            local_repo_path=local_repo_path,
            branch_name=branch_name,
            file_changes=fix.file_changes,
            commit_message=f"{ticket_key}: automated fix proposal\n\n{fix.explanation}",
        )

        mr_url = None
        if self.cfg["workflow"]["auto_open_merge_request"]:
            mr_url = self.gitlab.open_merge_request(
                project_path=repo_path_str,
                source_branch=branch_name,
                title=f"{ticket_key}: {ticket.summary}",
                description=(
                    f"Automated fix proposal from L3 Agent.\n\n"
                    f"**Root cause:** {rca.root_cause}\n\n"
                    f"**Fix explanation:** {fix.explanation}\n\n"
                    f"**Risk notes:** {fix.risk_notes}\n\n"
                    f"**Confidence:** {rca.confidence:.0%}\n\n"
                    f"_This MR was drafted automatically and requires human review before merge._"
                ),
            )

        comment = (
            f"[L3 Agent] Root cause identified (confidence {rca.confidence:.0%}).\n\n"
            f"Root cause: {rca.root_cause}\n\n"
            f"A draft fix has been proposed"
            + (f": {mr_url}" if mr_url else " (see agent logs)")
            + "\nThis requires human review before merging."
        )
        self.jira.post_comment(ticket_key, comment)
        self._audit("mr_opened", {"ticket": ticket_key, "mr_url": mr_url, "branch": branch_name})

        return RunResult(
            ticket_key,
            "fix_proposed",
            {"rca": asdict(rca), "fix_explanation": fix.explanation, "mr_url": mr_url, "branch": branch_name},
        )

    def _audit(self, event: str, data: dict) -> None:
        entry = {"ts": time.time(), "event": event, **data}
        with open(self.audit_log_path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")


def _resolve_env(value: str) -> str:
    import os
    if value and value.startswith("${") and value.endswith("}"):
        return os.environ.get(value[2:-1], "")
    return value
