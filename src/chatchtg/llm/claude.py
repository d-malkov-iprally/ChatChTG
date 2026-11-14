"""The Claude adapter: sends a Context to the Anthropic Messages API."""

from __future__ import annotations

import base64
from typing import Any

from chatchtg.context.types import Context, ImagePart
from chatchtg.llm.client import Completion

DEFAULT_MODEL = "claude-opus-5-5"
MAX_TOKENS = 16000
# Opus 5.5 defaults to medium; set it so every model runs at the same level.
EFFORT = "medium"
# If the model declines for policy reasons, the API re-runs the request on another model.
FALLBACK_BETA = "server-side-fallback-2026-07-01"
# No digits here: the eval scorer reads numbers from the answer.
REFUSAL_TEXT = "[no answer] The model declined to answer this request."


class ClaudeLLM:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self._api_key = api_key
        self._client = client

    def complete(self, context: Context) -> Completion:
        response = self._get_client().beta.messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            system=context.system,
            messages=[{"role": "user", "content": [_block(part) for part in context.parts]}],
            output_config={"effort": EFFORT},
            betas=[FALLBACK_BETA],
            fallbacks="default",
        )
        if response.stop_reason == "refusal":
            text = REFUSAL_TEXT
        else:
            text = "".join(block.text for block in response.content if block.type == "text")
        # response.model is the model that served the request, which a fallback can change.
        return Completion(text=text, model=response.model, usage=token_usage(response.usage))

    def _get_client(self) -> Any:
        # Created on first use, so building the adapter needs no credentials.
        if self._client is None:
            import anthropic

            if self._api_key:
                self._client = anthropic.Anthropic(api_key=self._api_key)
            else:
                # Reads ANTHROPIC_API_KEY, or the profile stored by `ant auth login`.
                self._client = anthropic.Anthropic()
        return self._client


def _block(part: Any) -> dict[str, Any]:
    if isinstance(part, ImagePart):
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": part.media_type,
                "data": base64.standard_b64encode(part.data).decode("ascii"),
            },
        }
    return {"type": "text", "text": part.text}


_USAGE_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
)


def token_usage(usage: Any) -> dict[str, int]:
    """The integer token counts of an API usage object."""
    counts = {name: getattr(usage, name, None) for name in _USAGE_FIELDS}
    return {name: value for name, value in counts.items() if isinstance(value, int)}
