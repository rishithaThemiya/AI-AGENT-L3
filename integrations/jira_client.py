"""
integrations/jira_client.py

Thin wrapper around the Jira REST API for the L3 agent's needs:
- fetch a ticket's full context (description, comments, attachments list)
- post the RCA / fix summary / explanation back as a comment
- transition ticket status (optional)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from jira import JIRA


@dataclass
class TicketContext:
    key: str
    summary: str
    description: str
    issue_type: str
    priority: str
    comments: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    components: list[str] = field(default_factory=list)

    def as_prompt_text(self) -> str:
        """Flatten the ticket into plain text for LLM consumption."""
        parts = [
            f"Ticket: {self.key}",
            f"Type: {self.issue_type} | Priority: {self.priority}",
            f"Components: {', '.join(self.components) or 'N/A'}",
            f"Labels: {', '.join(self.labels) or 'N/A'}",
            f"Summary: {self.summary}",
            f"Description:\n{self.description}",
        ]
        if self.comments:
            parts.append("Comments:\n" + "\n---\n".join(self.comments))
        return "\n\n".join(parts)


class JiraClient:
    def __init__(self, server: str, email: str, api_token: str):
        self.jira = JIRA(server=server, basic_auth=(email, api_token))

    def fetch_ticket(self, ticket_key: str) -> TicketContext:
        issue = self.jira.issue(ticket_key)
        fields = issue.fields

        comments = [c.body for c in getattr(fields.comment, "comments", [])]
        components = [c.name for c in getattr(fields, "components", [])]

        return TicketContext(
            key=issue.key,
            summary=fields.summary or "",
            description=fields.description or "",
            issue_type=fields.issuetype.name if fields.issuetype else "Unknown",
            priority=fields.priority.name if fields.priority else "Unknown",
            comments=comments,
            labels=list(getattr(fields, "labels", []) or []),
            components=components,
        )

    def post_comment(self, ticket_key: str, body: str) -> None:
        self.jira.add_comment(ticket_key, body)

    def transition_ticket(self, ticket_key: str, transition_name: str) -> None:
        issue = self.jira.issue(ticket_key)
        transitions = self.jira.transitions(issue)
        match = next((t for t in transitions if t["name"].lower() == transition_name.lower()), None)
        if match:
            self.jira.transition_issue(issue, match["id"])

    def get_open_tickets(self, project_key: str, jql_extra: str = "") -> list[str]:
        """Used for polling mode: returns ticket keys matching the project + status."""
        jql = f'project = "{project_key}" AND status = "Open" {jql_extra}'.strip()
        issues = self.jira.search_issues(jql, maxResults=50)
        return [issue.key for issue in issues]
