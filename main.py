"""
main.py

Local entry point. Two modes:

1. Index a repo (run this first, and periodically to keep RAG fresh):
   python main.py index --project L3

2. Process a single ticket end-to-end:
   python main.py handle --ticket L3-1234 --project L3

For continuous local testing, you can wrap `handle` in a polling loop
using JiraClient.get_open_tickets(), but starting with manual single-ticket
runs is safer while you're validating fix quality.
"""

import argparse
from rich.console import Console
from rich.pretty import pprint

from orchestrator import L3Orchestrator

console = Console()


def main():
    parser = argparse.ArgumentParser(description="L3 Production Support Agent")
    parser.add_argument("command", choices=["index", "handle"])
    parser.add_argument("--ticket", help="Jira ticket key, e.g. L3-1234")
    parser.add_argument("--project", required=True, help="Jira project key, e.g. L3")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    orchestrator = L3Orchestrator(config_path=args.config)

    if args.command == "index":
        console.print(f"[bold cyan]Indexing repo for project {args.project}...[/bold cyan]")
        count = orchestrator.index_project(args.project)
        console.print(f"[bold green]Indexed {count} code chunks.[/bold green]")

    elif args.command == "handle":
        if not args.ticket:
            console.print("[bold red]--ticket is required for 'handle'[/bold red]")
            return
        console.print(f"[bold cyan]Processing {args.ticket}...[/bold cyan]")
        result = orchestrator.handle_ticket(args.ticket, args.project)
        console.print(f"[bold green]Outcome: {result.outcome}[/bold green]")
        pprint(result.details)


if __name__ == "__main__":
    main()
