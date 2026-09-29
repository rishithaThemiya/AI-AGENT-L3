"""
agents/fix_agent.py

Drafts a fix for the identified root cause. IMPORTANT SAFETY NOTE:
this agent only produces proposed file contents + explanation. The
orchestrator is responsible for opening a draft Merge Request for
human review -- this code path never merges or force-pushes to a
protected branch on its own.
"""

from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from llm.provider import LLMProvider
from agents.rca_agent import RCAResult

SYSTEM_PROMPT = """You are an L3 production-support engineer drafting a code fix.
You will be given a root cause analysis and the full current content of the
suspect file(s). Produce a minimal, targeted fix -- do not refactor unrelated
code, do not change formatting/style beyond what's needed.

Respond with STRICT JSON only, schema:

{
  "explanation": "what you changed and why, 2-4 sentences",
  "risk_notes": "anything a human reviewer should double check",
  "file_changes": {
    "path/to/file.ext": "COMPLETE new file content with the fix applied"
  }
}

Rules:
- Only include files you actually changed in file_changes.
- Preserve the existing code style/conventions of each file.
- Never invent files that were not provided to you.
- If you are not confident enough to propose a safe fix, set file_changes
  to an empty object {} and explain why in "explanation".
"""


@dataclass
class FixProposal:
    explanation: str
    risk_notes: str
    file_changes: dict[str, str]


def run_fix_proposal(
    llm: LLMProvider,
    rca: RCAResult,
    current_file_contents: dict[str, str],
) -> FixProposal:
    files_block = "\n\n".join(
        f"--- CURRENT CONTENT: {path} ---\n{content}"
        for path, content in current_file_contents.items()
    )

    user_prompt = (
        f"Root cause: {rca.root_cause}\n"
        f"Evidence: {rca.evidence}\n\n"
        f"{files_block}"
    )

    response = llm.complete(system=SYSTEM_PROMPT, user=user_prompt, max_tokens=4000)
    data = json.loads(_strip_fences(response.text))

    return FixProposal(
        explanation=data["explanation"],
        risk_notes=data.get("risk_notes", ""),
        file_changes=data.get("file_changes", {}),
    )


def read_current_contents(repo_path: Path, relative_paths: list[str]) -> dict[str, str]:
    contents = {}
    for rel_path in relative_paths:
        full_path = repo_path / rel_path
        if full_path.exists():
            contents[rel_path] = full_path.read_text(encoding="utf-8", errors="ignore")
    return contents


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return text.strip()
