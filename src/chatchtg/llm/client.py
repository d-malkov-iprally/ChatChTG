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


def build_llm(provider: str, model: str | None = None) -> LLMClient:
    """Return the adapter for ``provider``. To add a provider, add a branch here.

    ``model`` picks the provider's model. None means the adapter's default.
    """
    if provider == "fake":
        from chatchtg.llm.fake import FakeLLM

        return FakeLLM()
    if provider == "claude":
        from chatchtg.llm.claude import DEFAULT_MODEL, ClaudeLLM

        return ClaudeLLM(model or DEFAULT_MODEL)
    raise ValueError(f"Unknown LLM provider {provider!r}. Available: claude, fake")
