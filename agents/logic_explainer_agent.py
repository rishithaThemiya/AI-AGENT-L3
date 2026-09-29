"""
agents/logic_explainer_agent.py

For tickets that ask to EXPLAIN or VERIFY existing code logic (not fix a
bug), this agent summarizes what the code actually does -- tables/fields
used, filters, calculation steps -- WITHOUT inventing a "root cause" or
implying something is wrong. This is the correct path for "review/verify
this logic for the client" type change requests, as opposed to bug RCA.
"""

from __future__ import annotations
from dataclasses import dataclass
from llm.provider import LLMProvider
from integrations.jira_client import TicketContext
from rag.retriever import RetrievedChunk
from agents.json_utils import parse_llm_json

SYSTEM_PROMPT = """You are an L3 support engineer explaining existing code logic
to a colleague who needs to verify it with a client. You are NOT looking for
bugs or problems -- only describe what the code actually does, factually.

Respond with STRICT JSON only, schema:

{
  "summary": "plain-language explanation of what this logic calculates/does",
  "relevant_functions": ["function_or_method_name", "..."],
  "tables_and_fields": "which DB tables/fields are used, if visible in the code",
  "filters_or_conditions": "any filters, date ranges, or conditions applied",
  "sql_query_if_present": "the actual SQL query text if one is visible in the chunks, else null",
  "files_reviewed": ["path/to/file1", "path/to/file2"],
  "caveats": "anything unclear, ambiguous, or that could not be fully confirmed from the code shown -- e.g. duplicate/near-duplicate logic across files, if genuinely present",
  "confidence": 0.0 to 1.0
}

Rules:
- Do NOT invent problems, bugs, or "root causes." This is a documentation/
  verification task, not a bug investigation.
- If you notice something like duplicated logic across files, mention it
  factually under "caveats" -- do not frame it as an "issue" or "root cause."
- If the retrieved code doesn't clearly show the requested logic, say so
  honestly in "caveats" and lower the confidence score, rather than guessing.
"""


@dataclass
class LogicExplanation:
    summary: str
    relevant_functions: list[str]
    tables_and_fields: str
    filters_or_conditions: str
    sql_query_if_present: str | None
    files_reviewed: list[str]
    caveats: str
    confidence: float


def run_logic_explanation(
    llm: LLMProvider,
    ticket: TicketContext,
    retrieved_chunks: list[RetrievedChunk],
) -> LogicExplanation:
    code_context = "\n\n".join(c.as_prompt_text() for c in retrieved_chunks)

    user_prompt = f"{ticket.as_prompt_text()}\n\n=== Retrieved code context ===\n{code_context}"

    response = llm.complete(system=SYSTEM_PROMPT, user=user_prompt, max_tokens=1500)
    data = parse_llm_json(response.text)

    return LogicExplanation(
        summary=data["summary"],
        relevant_functions=data.get("relevant_functions", []),
        tables_and_fields=data.get("tables_and_fields", ""),
        filters_or_conditions=data.get("filters_or_conditions", ""),
        sql_query_if_present=data.get("sql_query_if_present"),
        files_reviewed=data.get("files_reviewed", []),
        caveats=data.get("caveats", ""),
        confidence=float(data.get("confidence", 0.5)),
    )
