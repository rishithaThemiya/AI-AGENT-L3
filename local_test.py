"""
local_test.py

Standalone way to test whether the agent can correctly identify an issue,
WITHOUT needing Jira or GitLab API access. Use this when:
- you don't have Jira API permission yet
- you already have the target repo downloaded locally
- you just want to validate: "given this issue description, can the agent
  find the right code and explain the root cause?"

You manually paste the ticket details (title + description), point it at
a local repo folder, and it runs: Triage -> RAG retrieval -> RCA.
No fix is drafted and nothing is written back anywhere -- this is purely
a read-only "can it find the issue" test.

Usage:
    python local_test.py --repo /path/to/local/repo --ticket-file ticket.txt

    (or run without --ticket-file to paste the ticket interactively)

ticket.txt format (plain text), e.g.:
    Title: Discount not applied on checkout for VIP users
    Description: When a VIP user applies a discount code at checkout, the
    total price is not reduced. Regular users are unaffected. Started
    happening after last week's release.
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from llm.provider import build_llm_provider
from integrations.jira_client import TicketContext
from rag.indexer import CodebaseIndexer
from rag.retriever import CodebaseRetriever
from agents.triage_agent import run_triage
from agents.rca_agent import run_rca
from agents.logic_explainer_agent import run_logic_explanation

console = Console()


def read_ticket_interactively() -> TicketContext:
    console.print("[bold cyan]Paste the ticket title, then press Enter:[/bold cyan]")
    title = input("> ").strip()

    console.print(
        "[bold cyan]Paste the ticket description (end with a line containing only 'END'):[/bold cyan]"
    )
    lines = []
    while True:
        line = input()
        if line.strip() == "END":
            break
        lines.append(line)
    description = "\n".join(lines)

    return TicketContext(
        key="LOCAL-TEST",
        summary=title,
        description=description,
        issue_type="bug",
        priority="Unknown",
    )


def read_ticket_from_file(path: str) -> TicketContext:
    text = Path(path).read_text(encoding="utf-8")
    title = ""
    description_lines = []
    in_description = False

    for line in text.splitlines():
        if line.lower().startswith("title:"):
            title = line.split(":", 1)[1].strip()
        elif line.lower().startswith("description:"):
            in_description = True
            description_lines.append(line.split(":", 1)[1].strip())
        elif in_description:
            description_lines.append(line)

    if not title:
        # fallback: whole file is the description, first line is the title
        all_lines = text.splitlines()
        title = all_lines[0] if all_lines else "Untitled issue"
        description_lines = all_lines[1:]

    return TicketContext(
        key="LOCAL-TEST",
        summary=title,
        description="\n".join(description_lines).strip(),
        issue_type="bug",
        priority="Unknown",
    )


def get_git_history_if_available(repo_path: Path, file_paths: list[str], limit: int = 10) -> dict[str, list[dict]]:
    """Best-effort git history lookup. Returns empty dict if repo has no .git folder."""
    if not (repo_path / ".git").exists():
        return {}
    try:
        import git
        repo = git.Repo(repo_path)
        history = {}
        for rel_path in file_paths:
            commits = list(repo.iter_commits(paths=rel_path, max_count=limit))
            history[rel_path] = [
                {
                    "hexsha": c.hexsha[:8],
                    "author": c.author.name,
                    "date": c.committed_datetime.isoformat(),
                    "message": c.message.strip(),
                }
                for c in commits
            ]
        return history
    except Exception as e:
        console.print(f"[yellow]Could not read git history: {e}[/yellow]")
        return {}


def main():
    parser = argparse.ArgumentParser(description="Local-only agent test: manual ticket + local repo path")
    parser.add_argument("--repo", required=True, help="Path to the local repo folder you already downloaded")
    parser.add_argument("--ticket-file", help="Path to a text file with 'Title:' and 'Description:' lines")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config.yaml (only llm + rag sections are used)")
    parser.add_argument("--reindex", action="store_true", help="Force re-indexing the repo even if already indexed")
    args = parser.parse_args()

    repo_path = Path(args.repo).expanduser().resolve()
    if not repo_path.exists():
        console.print(f"[bold red]Repo path does not exist: {repo_path}[/bold red]")
        sys.exit(1)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    console.print("[bold cyan]Loading LLM provider...[/bold cyan]")
    llm = build_llm_provider(cfg["llm"])

    console.print("[bold cyan]Loading indexer/retriever...[/bold cyan]")
    indexer = CodebaseIndexer(
        vector_db_path=cfg["rag"]["vector_db_path"],
        collection_name=cfg["rag"]["collection_name"],
        embedding_model=cfg["rag"]["embedding_model"],
        chunk_lines=cfg["rag"]["chunk_lines"],
        chunk_overlap_lines=cfg["rag"]["chunk_overlap_lines"],
        include_extensions=cfg["rag"]["include_extensions"],
        exclude_dirs=cfg["rag"]["exclude_dirs"],
    )
    retriever = CodebaseRetriever(
        vector_db_path=cfg["rag"]["vector_db_path"],
        collection_name=cfg["rag"]["collection_name"],
        embedding_model=cfg["rag"]["embedding_model"],
    )

    repo_label = str(repo_path)  # use the local path itself as the namespace label

    # Only re-index if the user explicitly asked for it, or if this repo
    # hasn't been indexed before (empty collection for this repo_label).
    # This avoids re-embedding thousands of chunks on every single run.
    existing_count = indexer.collection.count()
    already_indexed = existing_count > 0

    if args.reindex or not already_indexed:
        console.print(f"[bold cyan]Indexing {repo_path} ...[/bold cyan] (this may take a while on first run)")
        count = indexer.index_repo(repo_path, repo_label=repo_label)
        console.print(f"[green]Indexed {count} code chunks from local repo.[/green]")
    else:
        console.print(f"[green]Using existing index ({existing_count} chunks total in store). "
                       f"Pass --reindex to force a refresh.[/green]")

    # 1. Get the ticket
    if args.ticket_file:
        ticket = read_ticket_from_file(args.ticket_file)
    else:
        ticket = read_ticket_interactively()

    console.print(Panel(ticket.as_prompt_text(), title="Ticket", border_style="blue"))

    # 2. Triage
    console.print("[bold cyan]Running triage...[/bold cyan]")
    triage = run_triage(llm, ticket)
    console.print(Panel(
        f"issue_type: {triage.issue_type}\n"
        f"is_code_issue: {triage.is_code_issue}\n"
        f"confidence: {triage.confidence:.0%}\n"
        f"search_query: {triage.search_query}\n\n"
        f"reasoning: {triage.reasoning}",
        title="Triage Result", border_style="magenta",
    ))

    if not triage.is_code_issue:
        console.print("[bold yellow]Agent concluded this is NOT a code issue. Stopping here "
                       "(this matches the 'explain, don't fix' branch of the real flow).[/bold yellow]")
        return

    # 3. Retrieve relevant code
    console.print("[bold cyan]Searching codebase...[/bold cyan]")
    chunks = retriever.search(query=triage.search_query, repo_label=repo_label, top_k=cfg["rag"]["top_k"])

    if not chunks:
        console.print("[bold red]No relevant code chunks found. Try adjusting rag.include_extensions "
                       "in config.yaml to match this repo's language, or check the repo actually has code files.[/bold red]")
        return

    console.print(f"[green]Found {len(chunks)} relevant chunks:[/green]")
    for c in chunks:
        console.print(f"  - {c.file_path} (lines {c.start_line}-{c.end_line}, distance {c.distance:.3f})")

    # 4. Git history (best-effort, works only if repo_path is an actual git repo)
    suspect_files = sorted({c.file_path for c in chunks})
    commit_history = get_git_history_if_available(repo_path, suspect_files)

    # 5a. If this is a genuine bug, run RCA (root cause hypothesis).
    # 5b. If this is a "verify/explain existing logic" change request,
    #     run the logic explainer instead -- RCA would otherwise force-fit
    #     a "root cause" onto code that isn't actually broken.
    if triage.issue_type == "bug":
        console.print("[bold cyan]Running root cause analysis...[/bold cyan]")
        rca = run_rca(llm, ticket, chunks, commit_history)

        console.print(Panel(
            Markdown(
                f"**Root cause:** {rca.root_cause}\n\n"
                f"**Evidence:** {rca.evidence}\n\n"
                f"**Suspect files:** {', '.join(rca.suspect_files) or 'none identified'}\n\n"
                f"**Confidence:** {rca.confidence:.0%}\n\n"
                f"**Likely introduced by commit:** {rca.likely_introduced_by_commit or 'unclear'}"
            ),
            title="Root Cause Analysis", border_style="green",
        ))

        min_conf = cfg["workflow"]["min_confidence_to_propose_fix"]
        if rca.confidence >= min_conf:
            console.print(f"[bold green]Confidence ({rca.confidence:.0%}) is above the fix threshold "
                           f"({min_conf:.0%}) -- in the full flow, the agent would now draft a fix.[/bold green]")
        else:
            console.print(f"[bold yellow]Confidence ({rca.confidence:.0%}) is below the fix threshold "
                           f"({min_conf:.0%}) -- agent would only report RCA, not attempt a fix.[/bold yellow]")

    else:
        console.print("[bold cyan]Explaining existing logic (not a bug -- no root cause will be invented)...[/bold cyan]")
        explanation = run_logic_explanation(llm, ticket, chunks)

        sql_block = f"\n\n**SQL query found:**\n```\n{explanation.sql_query_if_present}\n```" if explanation.sql_query_if_present else ""

        console.print(Panel(
            Markdown(
                f"**Summary:** {explanation.summary}\n\n"
                f"**Relevant functions:** {', '.join(explanation.relevant_functions) or 'none identified'}\n\n"
                f"**Tables/fields used:** {explanation.tables_and_fields or 'not clearly visible in retrieved code'}\n\n"
                f"**Filters/conditions:** {explanation.filters_or_conditions or 'none identified'}\n\n"
                f"**Files reviewed:** {', '.join(explanation.files_reviewed) or 'none'}\n\n"
                f"**Caveats:** {explanation.caveats or 'none'}\n\n"
                f"**Confidence:** {explanation.confidence:.0%}"
                f"{sql_block}"
            ),
            title="Logic Explanation (for client verification)", border_style="green",
        ))


if __name__ == "__main__":
    main()
