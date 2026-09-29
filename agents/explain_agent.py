"""
agents/explain_agent.py

For tickets that are NOT code issues (config, infra, data, user error,
duplicate), drafts a clear explanation to post back on the Jira ticket
instead of a code fix.
"""

from __future__ import annotations
from llm.provider import LLMProvider
from integrations.jira_client import TicketContext
from agents.triage_agent import TriageResult

SYSTEM_PROMPT = """You are an L3 production-support engineer replying to a Jira
ticket that turned out NOT to be a code issue. Write a clear, professional
comment (plain text, no markdown headers) explaining:
1. What the actual issue appears to be
2. Why it isn't a code/application bug
3. What the reporter or another team should do next

Keep it concise -- 4-8 sentences. This will be posted directly as a Jira comment.
"""


def run_explanation(llm: LLMProvider, ticket: TicketContext, triage: TriageResult) -> str:
    user_prompt = (
        f"{ticket.as_prompt_text()}\n\n"
        f"Triage classification: {triage.issue_type}\n"
        f"Triage reasoning: {triage.reasoning}"
    )
    response = llm.complete(system=SYSTEM_PROMPT, user=user_prompt, max_tokens=600)
    return response.text.strip()
