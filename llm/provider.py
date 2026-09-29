"""
llm/provider.py

Provider-agnostic LLM abstraction so agents can call `llm.complete(...)`
without caring whether it's Claude, OpenAI, or a local Ollama model.

Swap providers by changing config.yaml -> llm.provider, no code changes
needed elsewhere in the agent.
"""

from __future__ import annotations
import abc
import os
from dataclasses import dataclass


@dataclass
class LLMResponse:
    text: str
    raw: dict | None = None


class LLMProvider(abc.ABC):
    """Common interface every provider must implement."""

    @abc.abstractmethod
    def complete(self, system: str, user: str, max_tokens: int = 2000) -> LLMResponse:
        ...


class AnthropicProvider(LLMProvider):
    def __init__(self, model: str, api_key: str | None = None):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model

    def complete(self, system: str, user: str, max_tokens: int = 2000) -> LLMResponse:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(block.text for block in resp.content if block.type == "text")
        return LLMResponse(text=text, raw=resp.model_dump())


class OpenAIProvider(LLMProvider):
    def __init__(self, model: str, api_key: str | None = None):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.model = model

    def complete(self, system: str, user: str, max_tokens: int = 2000) -> LLMResponse:
        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        text = resp.choices[0].message.content or ""
        return LLMResponse(text=text, raw=resp.model_dump())


class OllamaProvider(LLMProvider):
    """Local model via Ollama -- useful for offline dev/testing without API cost."""

    def __init__(self, model: str, host: str = "http://localhost:11434", num_ctx: int = 8192):
        import ollama
        self.client = ollama.Client(host=host)
        self.model = model
        self.num_ctx = num_ctx

    def complete(self, system: str, user: str, max_tokens: int = 2000) -> LLMResponse:
        resp = self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            options={"num_predict": max_tokens, "num_ctx": self.num_ctx},
        )
        text = resp["message"]["content"]
        return LLMResponse(text=text, raw=resp)


def build_llm_provider(cfg: dict, override_provider: str | None = None) -> LLMProvider:
    """
    Factory: reads the 'llm' section of config.yaml and returns the
    configured provider. Pass override_provider to force a specific
    backend at call time (e.g. testing with Ollama while default is Anthropic).
    """
    provider_name = override_provider or cfg["provider"]

    if provider_name == "anthropic":
        sub = cfg["anthropic"]
        return AnthropicProvider(model=sub["model"], api_key=_resolve_env(sub.get("api_key")))
    elif provider_name == "openai":
        sub = cfg["openai"]
        return OpenAIProvider(model=sub["model"], api_key=_resolve_env(sub.get("api_key")))
    elif provider_name == "ollama":
        sub = cfg["ollama"]
        return OllamaProvider(
            model=sub["model"],
            host=sub.get("host", "http://localhost:11434"),
            num_ctx=sub.get("num_ctx", 8192),
        )
    else:
        raise ValueError(f"Unknown LLM provider: {provider_name}")


def _resolve_env(value: str | None) -> str | None:
    """Supports '${ENV_VAR}' syntax in config.yaml."""
    if value and value.startswith("${") and value.endswith("}"):
        return os.environ.get(value[2:-1])
    return value
