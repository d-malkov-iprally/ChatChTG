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
    modal_endpoint: str | None = None,
) -> LLMClient:
    """Return the adapter for ``provider``. To add a provider, add a branch here.

    ``model`` picks the provider's model. None means the adapter's default.
    ``api_key`` is the provider's key: for ``modal``, the proxy token. None means the
    provider SDK's own credential lookup; Modal has none, so its adapter then fails on the
    first request.
    ``modal_endpoint`` is the hostname of the Modal endpoint that the ``modal`` adapter calls.
    """
    if provider == "fake":
        from chatchtg.llm.fake import FakeLLM

        return FakeLLM()
    if provider == "claude":
        from chatchtg.llm.claude import DEFAULT_MODEL, ClaudeLLM

        return ClaudeLLM(model or DEFAULT_MODEL, api_key=api_key)
    if provider == "modal":
        from chatchtg.llm import modal

        return modal.ModalLLM(
            model or modal.DEFAULT_MODEL, endpoint=modal_endpoint, proxy_token=api_key
        )
    raise ValueError(f"Unknown LLM provider {provider!r}. Available: claude, fake, modal")
