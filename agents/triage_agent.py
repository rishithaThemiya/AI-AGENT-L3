"""
agents/triage_agent.py

First step in the flow: reads the Jira ticket and decides:
- is this a bug or a change request (CR)?
- is it actually a code issue, or something else (config, data, infra,
  user error, duplicate)?

Returns structured JSON so the orchestrator can branch reliably instead
of parsing free text.
"""

from __future__ import annotations
from dataclasses import dataclass
from llm.provider import LLMProvider
from integrations.jira_client import TicketContext
from agents.json_utils import parse_llm_json

SYSTEM_PROMPT = """You are an L3 production-support triage assistant.
Given a Jira ticket, classify it and respond with STRICT JSON only, no
markdown fences, no extra commentary. Schema:

{
  "issue_type": "bug" | "change_request" | "not_code_issue" | "duplicate_or_unclear",
  "is_code_issue": true | false,
  "reasoning": "short explanation, 2-3 sentences",
  "search_query": "a concise natural-language query to find the relevant code, e.g. 'null pointer in payment validation when discount code applied'",
  "confidence": 0.0 to 1.0
}

Guidance:
- "bug" means something is BROKEN -- actual behavior differs from expected
  behavior, an error occurs, or output is wrong.
- "change_request" covers: requests to review, verify, explain, or document
  EXISTING logic (nothing is reported as broken); requests for new features;
  requests to modify behavior on purpose. If the ticket does not describe
  any incorrect/broken behavior, it is a change_request, not a bug, even if
  it requires reading code.
- "not_code_issue" covers: user permission errors, environment/infra issues,
  bad data, network problems, third-party outages -- anything not fixable
  by changing this codebase.
- "duplicate_or_unclear" if the ticket lacks enough info to act on.
- search_query should describe the SYMPTOM/behavior, not Jira jargon,
  since it will be used to search source code semantically.
"""


@dataclass
class TriageResult:
    issue_type: str
    is_code_issue: bool
    reasoning: str
    search_query: str
    confidence: float


def run_triage(llm: LLMProvider, ticket: TicketContext) -> TriageResult:
    user_prompt = f"Jira ticket to triage:\n\n{ticket.as_prompt_text()}"
    response = llm.complete(system=SYSTEM_PROMPT, user=user_prompt, max_tokens=500)

    data = parse_llm_json(response.text)
    return TriageResult(
        issue_type=data["issue_type"],
        is_code_issue=data["is_code_issue"],
        reasoning=data["reasoning"],
        search_query=data["search_query"],
        confidence=float(data["confidence"]),
    )

