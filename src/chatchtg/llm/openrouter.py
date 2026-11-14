"""The OpenRouter adapter: sends a Context to OpenRouter's Chat Completions API."""

from __future__ import annotations

import base64
import json
import ssl
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

import certifi

from chatchtg.context.types import Context, ImagePart
from chatchtg.llm.client import Completion

URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "moonshotai/kimi-k3.1-pro"
# Reasoning tokens count against this limit. Kimi K3.1 Pro reasons at its highest level by
# default, and the adapter leaves that default.
MAX_TOKENS = 32000
# A reasoning model can take minutes for one answer.
TIMEOUT_SECONDS = 600
# No digits in these: the eval scorer reads numbers from the answer.
REFUSAL_TEXT = "[no answer] The model declined to answer this request."
NO_BUDGET_TEXT = "[no answer] The model reached its output limit before it gave an answer."


class OpenRouterError(RuntimeError):
    """OpenRouter did not return a completion."""


class OpenRouterLLM:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        provider: str | None = None,
        post: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        """``provider`` is the OpenRouter slug of the only provider to use. None means
        OpenRouter picks one for each request. ``post`` replaces the HTTP call: it gets the
        request body and returns the response body.
        """
        self.model = model
        self.provider = provider
        self._api_key = api_key
        self._post = post or self._http_post

    def complete(self, context: Context) -> Completion:
        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": MAX_TOKENS,
            "messages": [
                {"role": "system", "content": context.system},
                {"role": "user", "content": [_block(part) for part in context.parts]},
            ],
        }
        if self.provider:
            body["provider"] = {"only": [self.provider]}
        response = self._post(body)

        # OpenRouter reports some failures with status 200: in the body, or in the choice.
        choices = response.get("choices") or [{}]
        error = response.get("error") or choices[0].get("error")
        if error or not response.get("choices"):
            raise OpenRouterError(_error_message(error, response))
        choice = choices[0]
        text = (choice.get("message") or {}).get("content") or ""
        if choice.get("finish_reason") == "content_filter":
            text = REFUSAL_TEXT
        elif not text and choice.get("finish_reason") == "length":
            text = NO_BUDGET_TEXT
        # The requested model and not response["model"], which carries a release date:
        # the requested name is the key of the price in pricing.py.
        return Completion(text=text, model=self.model, usage=token_usage(response.get("usage")))

    def _http_post(self, body: dict[str, Any]) -> dict[str, Any]:
        # The key is checked on first use, so building the adapter needs no credentials.
        if not self._api_key:
            raise OpenRouterError(
                "The openrouter provider needs a key. Set CHATCHTG_OPENROUTER_API_KEY."
            )
        request = urllib.request.Request(
            URL,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        # The Python of python.org on macOS has no root certificates of its own.
        certificates = ssl.create_default_context(cafile=certifi.where())
        try:
            with urllib.request.urlopen(
                request, timeout=TIMEOUT_SECONDS, context=certificates
            ) as reply:
                raw = reply.read()
        except urllib.error.HTTPError as error:
            raise OpenRouterError(
                f"OpenRouter answered with status {error.code}: {_http_error_detail(error)}"
            ) from None
        except (urllib.error.URLError, TimeoutError) as error:
            raise OpenRouterError(f"OpenRouter could not be reached: {error}") from None
        try:
            return json.loads(raw)
        except ValueError:
            raise OpenRouterError("OpenRouter sent a response that is not JSON.") from None


def _block(part: Any) -> dict[str, Any]:
    if isinstance(part, ImagePart):
        data = base64.standard_b64encode(part.data).decode("ascii")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{part.media_type};base64,{data}"},
        }
    return {"type": "text", "text": part.text}


def _error_message(error: Any, response: dict[str, Any]) -> str:
    if isinstance(error, dict):
        return f"OpenRouter reported an error: {error.get('message')} (code {error.get('code')})"
    return f"OpenRouter sent no completion (generation {response.get('id')})."


def _http_error_detail(error: urllib.error.HTTPError) -> str:
    """The message of an error response, or the reason phrase when the body has none."""
    try:
        message = json.loads(error.read())["error"]["message"]
    except (ValueError, KeyError, TypeError):
        return str(error.reason)
    return str(message)


# OpenRouter's names for the token counts, and the names that pricing.py reads.
_USAGE_FIELDS = {"prompt_tokens": "input_tokens", "completion_tokens": "output_tokens"}


def token_usage(usage: Any) -> dict[str, int]:
    """The integer token counts of a usage object. Reasoning tokens are output tokens."""
    if not isinstance(usage, dict):
        return {}
    counts = {ours: usage.get(theirs) for theirs, ours in _USAGE_FIELDS.items()}
    return {name: value for name, value in counts.items() if isinstance(value, int)}
