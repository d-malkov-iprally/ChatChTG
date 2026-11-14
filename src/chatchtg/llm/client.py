"""The LLM interface. Each provider is one adapter that implements LLMClient."""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, Field

from chatchtg.context.types import Context


class Completion(BaseModel):
    text: str
    model: str
    usage: dict[str, int] = Field(default_factory=dict)


class LLMClient(Protocol):
    def complete(self, context: Context) -> Completion: ...


def build_llm(
    provider: str,
    model: str | None = None,
    api_key: str | None = None,
    openrouter_provider: str | None = None,
) -> LLMClient:
    """Return the adapter for ``provider``. To add a provider, add a branch here.

    ``model`` picks the provider's model. None means the adapter's default.
    ``api_key`` is the provider's key. None means the provider SDK's own credential lookup;
    OpenRouter has none, so its adapter then fails on the first request.
    ``openrouter_provider`` is the only provider that OpenRouter may route to. None means
    OpenRouter picks.
    """
    if provider == "fake":
        from chatchtg.llm.fake import FakeLLM

        return FakeLLM()
    if provider == "claude":
        from chatchtg.llm.claude import DEFAULT_MODEL, ClaudeLLM

        return ClaudeLLM(model or DEFAULT_MODEL, api_key=api_key)
    if provider == "openrouter":
        from chatchtg.llm import openrouter

        return openrouter.OpenRouterLLM(
            model or openrouter.DEFAULT_MODEL, api_key=api_key, provider=openrouter_provider
        )
    raise ValueError(f"Unknown LLM provider {provider!r}. Available: claude, fake, openrouter")
