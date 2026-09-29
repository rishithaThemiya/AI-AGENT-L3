"""
agents/rca_agent.py

Combines: ticket context + retrieved code chunks + recent git commit
history for the suspect files, and asks the LLM to produce a root cause
hypothesis. This works the same regardless of the repo's language --
the LLM reads whatever source code is in the chunks.
"""

from __future__ import annotations
import json
from dataclasses import dataclass
from llm.provider import LLMProvider
from integrations.jira_client import TicketContext
from rag.retriever import RetrievedChunk
from agents.json_utils import parse_llm_json

SYSTEM_PROMPT = """You are an L3 production-support root-cause-analysis assistant.
You will be given a Jira ticket, relevant source code chunks retrieved from
the codebase, and recent commit history for those files.

Respond with STRICT JSON only, schema:

{
  "root_cause": "clear technical explanation of what is actually wrong",
  "evidence": "which specific code/lines/commits support this conclusion",
  "suspect_files": ["path/to/file1", "path/to/file2"],
  "confidence": 0.0 to 1.0,
  "likely_introduced_by_commit": "commit hash or null if unclear"
}

Be honest about uncertainty -- if the evidence is thin, say so and lower
the confidence score rather than guessing confidently.
"""


@dataclass
class RCAResult:
    root_cause: str
    evidence: str
    suspect_files: list[str]
    confidence: float
    likely_introduced_by_commit: str | None


def run_rca(
    llm: LLMProvider,
    ticket: TicketContext,
    retrieved_chunks: list[RetrievedChunk],
    commit_history: dict[str, list[dict]],
) -> RCAResult:
    code_context = "\n\n".join(c.as_prompt_text() for c in retrieved_chunks)

    history_text = ""
    for file_path, commits in commit_history.items():
        history_text += f"\nRecent commits touching {file_path}:\n"
        for c in commits:
            history_text += f"  - {c['hexsha']} ({c['date']}, {c['author']}): {c['message']}\n"

    user_prompt = (
        f"{ticket.as_prompt_text()}\n\n"
        f"=== Retrieved code context ===\n{code_context}\n\n"
        f"=== Git history ===\n{history_text}"
    )

    response = llm.complete(system=SYSTEM_PROMPT, user=user_prompt, max_tokens=1500)
    data = parse_llm_json(response.text)

    return RCAResult(
        root_cause=data["root_cause"],
        evidence=data["evidence"],
        suspect_files=data.get("suspect_files", []),
        confidence=float(data["confidence"]),
        likely_introduced_by_commit=data.get("likely_introduced_by_commit"),
    )
