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


def build_llm(provider: str) -> LLMClient:
    """Return the adapter for ``provider``. To add a provider, add a branch here."""
    if provider == "fake":
        from chatchtg.llm.fake import FakeLLM

        return FakeLLM()
    raise ValueError(f"Unknown LLM provider {provider!r}. Available: fake")
