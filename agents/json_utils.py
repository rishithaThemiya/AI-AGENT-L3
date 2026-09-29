"""
agents/json_utils.py

Shared helper for parsing JSON out of LLM responses. Local/smaller models
(and even larger ones, occasionally) sometimes emit invalid JSON escapes --
most commonly when a Windows-style file path like "models\\Foo.php" gets
echoed back with a single backslash, which isn't a valid JSON escape
sequence (only \\n, \\t, \\", \\\\, etc. are valid).

This sanitizes those before parsing instead of crashing the whole run.
"""

from __future__ import annotations
import json
import re

_VALID_ESCAPES = set('"\\/bfnrtu')


def strip_fences(text: str) -> str:
    """Strip ```json ... ``` fences if the model wrapped its output in them."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return text.strip()


def _escape_invalid_backslashes(text: str) -> str:
    """
    Walk the string and double-up any backslash that isn't part of a
    valid JSON escape sequence, so e.g. 'models\\Foo.php' becomes
    'models\\\\Foo.php' (a literal backslash) instead of an invalid escape.
    """
    def replace(match: re.Match) -> str:
        next_char = match.group(1)
        if next_char in _VALID_ESCAPES:
            return match.group(0)
        return "\\\\" + next_char

    return re.sub(r"\\(.)", replace, text)


def parse_llm_json(raw_text: str) -> dict:
    """
    Robustly parse a JSON object out of an LLM response. Tries a normal
    parse first; if that fails specifically due to invalid escapes, retries
    after sanitizing backslashes. Raises the original error if the text
    still isn't valid JSON after that (so callers can see the real problem).
    """
    text = strip_fences(raw_text)
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        if "escape" in str(e).lower():
            sanitized = _escape_invalid_backslashes(text)
            return json.loads(sanitized)
        raise
