# L3 Production Support Agent

An agent skeleton for L3 production support: reads a Jira ticket, decides
whether it's a genuine code issue, and either drafts a fix (as a GitLab
draft Merge Request, never auto-merged) or explains why it isn't a code
issue -- posting the result back onto the Jira ticket either way.

Built to be **language-agnostic** (works across Python/Java/Go/PHP/etc.
repos without changes) and **LLM-provider-agnostic** (Claude, OpenAI, or a
local Ollama model -- swap via config, no code changes).

## How it works

```
Jira ticket
  -> Triage agent          (code issue? bug vs CR vs not-code)
      -> not code issue    -> Explanation agent -> post Jira comment -> done
      -> code issue:
          -> RAG retrieval  (semantic search over the codebase)
          -> Git history    (recent commits on suspect files)
          -> RCA agent      (root cause hypothesis + confidence score)
          -> confidence >= threshold?
              -> yes: Fix agent -> branch + draft Merge Request (human review required)
              -> no:  post RCA-only comment, flagged for human investigation
```

**Safety design:** the agent never merges code on its own. It only ever
opens a *draft* Merge Request. A human always reviews and merges.

## Setup

```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Copy the config template and fill in your real values:

```bash
cp config/config.example.yaml config/config.yaml
```

Set secrets as environment variables (referenced via `${VAR_NAME}` in
config.yaml), e.g. in a `.env` file or your shell:

```bash
export JIRA_API_TOKEN="..."
export GITLAB_TOKEN="..."
export ANTHROPIC_API_KEY="..."      # if using Claude
export OPENAI_API_KEY="..."         # if using OpenAI
# Ollama needs no key, just `ollama serve` running locally
```

Edit `config/config.yaml`:
- `jira.project_key` / `gitlab.project_map` -- map your Jira project to the
  matching GitLab repo path.
- `llm.provider` -- `anthropic`, `openai`, or `ollama`. Since you want to
  test both a paid API and a local model, just flip this value between
  runs to compare quality/cost.
- `rag.include_extensions` -- adjust per repo language (defaults cover most
  common languages already).

## Usage

**1. Index the target repo** (do this first, and re-run periodically or
after major merges to keep the RAG index fresh):

```bash
python main.py index --project L3
```

**2. Process a ticket end-to-end:**

```bash
python main.py handle --ticket L3-1234 --project L3
```

This will:
- fetch the ticket from Jira
- triage it
- if it's a code issue: search the codebase, pull git history, run RCA
- if confident enough: draft a fix, push a branch, open a draft MR
- post the outcome as a Jira comment
- log every step to `logs/agent_audit.jsonl` for traceability

## Testing with different LLM providers

Since you want to test both hosted APIs and a local model:

```bash
# In config.yaml set llm.provider: "anthropic", or override per-run
# by editing the config, or extend main.py with a --llm-provider flag
# using build_llm_provider(cfg["llm"], override_provider="ollama")
```

Ollama example (fully offline, no API cost):

```bash
ollama pull qwen2.5-coder:14b
ollama serve
```

Then set `llm.provider: "ollama"` in config.yaml and re-run.

## Testing without Jira/GitLab access

If you don't have Jira API permission yet, or just want to validate that
the agent can correctly identify an issue before wiring up the full flow,
use `local_test.py`. It skips Jira and GitLab entirely -- you manually
type/paste the ticket details and point it at a repo you've already
downloaded locally. It runs Triage -> RAG retrieval -> RCA and prints the
result. It does not write anything anywhere (no Jira comment, no branch,
no MR) -- purely read-only, so it's safe to run against a real repo.

You still need `config/config.yaml` for this (only the `llm` and `rag`
sections are actually used -- the `jira:`/`gitlab:` sections can stay as
placeholders, they're ignored by this script).

```bash
cp config/config.example.yaml config/config.yaml
# fill in llm section (Anthropic/OpenAI/Ollama) and check rag.include_extensions
# matches the language of the repo you're testing against

# Option A: ticket details in a file
python local_test.py --repo /path/to/your/local/repo --ticket-file example_ticket.txt

# Option B: paste the ticket interactively
python local_test.py --repo /path/to/your/local/repo
# (paste title, Enter, then paste description, finish with a line: END)
```

Example `ticket.txt` format (see `example_ticket.txt`):
```
Title: Discount not applied on checkout for VIP users
Description: When a VIP user applies a discount code at checkout, the
total price is not reduced. Regular users are unaffected. Started
happening after last week's release.
```

What to look for in the output:
- **Triage result** -- did it correctly say `is_code_issue: true` for a
  real bug, and pick a sensible `search_query`?
- **Retrieved chunks** -- are the file paths it found actually relevant to
  the bug? If not, try adjusting `rag.chunk_lines`/`top_k`, or the issue
  description may need more technical detail.
- **RCA result** -- does the root cause explanation actually match what
  you know is wrong in the code? Check the confidence score against your
  own judgment.

Once this consistently finds the right code and gives sensible RCAs, wire
up Jira (once you get API access) and GitLab for the full automated flow.

## Recommended rollout order

1. Run `index` + `handle` against a **test/staging repo first**, not
   production, until you trust the fix quality.
2. Start with `min_confidence_to_propose_fix` set high (e.g. 0.8) so the
   agent only attempts fixes it's very sure about -- lower it gradually as
   you build trust.
3. Keep `auto_open_merge_request: true` but **never** give the agent's
   GitLab token merge permissions -- only permission to push branches and
   open MRs.
4. Review the audit log (`logs/agent_audit.jsonl`) regularly to see what
   the agent concluded at each step, especially for tickets marked
   `rca_only` (it wasn't confident enough to fix) or `not_code_issue`.

## Extending this skeleton

- **Polling mode:** wrap `handle_ticket` in a loop using
  `JiraClient.get_open_tickets()` to process new tickets automatically
  instead of running manually per ticket.
- **Test execution:** add a step after `fix_agent` that runs
  `workflow.test_command` (from config) inside the repo and includes
  pass/fail in the MR description before opening it.
- **LangGraph:** the current orchestrator is a plain Python state machine
  for readability. If you want visual graph debugging, retries, or
  human-in-the-loop interrupts, port `orchestrator.py`'s logic into a
  LangGraph `StateGraph` -- the agent functions in `agents/` don't need to
  change.
- **Per-language chunking:** the RAG indexer currently chunks by line
  windows (language-agnostic). If a specific repo would benefit from
  function-level chunking, add a `tree-sitter` based chunker for that
  file extension in `rag/indexer.py`.
