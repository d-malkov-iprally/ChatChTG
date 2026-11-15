"""The Modal adapter: sends a Context to a Modal Shared Endpoint (Chat Completions API)."""

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

# Modal serves every Shared Endpoint of a workspace through this address. The request's
# ``model`` field names the endpoint.
URL = "https://inference.us-west.modal.direct/v1/chat/completions"
# The model that the endpoint is expected to serve. It is the name in the results and the
# key of the price in pricing.py; it is not sent to Modal.
DEFAULT_MODEL = "moonshotai/Kimi-K3.1-Pro"
# Reasoning tokens count against this limit, and the adapter leaves the model's own
# reasoning level.
MAX_TOKENS = 32000
# A reasoning model can take minutes for one answer.
TIMEOUT_SECONDS = 600
# No digits here: the eval scorer reads numbers from the answer.
NO_BUDGET_TEXT = "[no answer] The model reached its output limit before it gave an answer."


class ModalError(RuntimeError):
    """Modal did not return a completion."""


class ModalLLM:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        endpoint: str | None = None,
        proxy_token: str | None = None,
        post: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        """``endpoint`` is the hostname of the Shared Endpoint, as the Modal dashboard shows
        it. ``proxy_token`` is a workspace proxy token in its combined form,
        ``wk-<id>.ws-<secret>``. ``post`` replaces the HTTP call: it gets the request body
        and returns the response body.
        """
        self.model = model
        self.endpoint = endpoint
        self._proxy_token = proxy_token
        self._post = post or self._http_post

    def complete(self, context: Context) -> Completion:
        # Checked on first use, so building the adapter needs no endpoint and no token.
        if not self.endpoint:
            raise ModalError(
                "The modal provider needs the hostname of a Modal endpoint. "
                "Set CHATCHTG_MODAL_ENDPOINT."
            )
        response = self._post(
            {
                "model": self.endpoint,
                "max_tokens": MAX_TOKENS,
                "messages": [
                    {"role": "system", "content": context.system},
                    {"role": "user", "content": [_block(part) for part in context.parts]},
                ],
            }
        )
        error = response.get("error")
        if error or not response.get("choices"):
            raise ModalError(f"Modal sent no completion: {_error_text(error) or response}")
        choice = response["choices"][0]
        text = (choice.get("message") or {}).get("content") or ""
        if not text and choice.get("finish_reason") == "length":
            text = NO_BUDGET_TEXT
        return Completion(text=text, model=self.model, usage=token_usage(response.get("usage")))

    def _http_post(self, body: dict[str, Any]) -> dict[str, Any]:
        if not self._proxy_token:
            raise ModalError(
                "The modal provider needs a proxy token. Set CHATCHTG_MODAL_PROXY_TOKEN."
            )
        request = urllib.request.Request(
            URL,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._proxy_token}",
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
            raise ModalError(
                f"Modal answered with status {error.code}: {_http_error_detail(error)}"
            ) from None
        except (urllib.error.URLError, TimeoutError) as error:
            raise ModalError(f"Modal could not be reached: {error}") from None
        try:
            parsed = json.loads(raw)
        except ValueError:
            parsed = None
        if not isinstance(parsed, dict):
            raise ModalError("Modal sent a response that is not a JSON object.")
        return parsed


def _block(part: Any) -> dict[str, Any]:
    if isinstance(part, ImagePart):
        data = base64.standard_b64encode(part.data).decode("ascii")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{part.media_type};base64,{data}"},
        }
    return {"type": "text", "text": part.text}


def _error_text(error: Any) -> str:
    """The message of an ``error`` field, which is an object or a plain string."""
    if isinstance(error, dict):
        return str(error.get("message") or error)
    return str(error) if error else ""


def _http_error_detail(error: urllib.error.HTTPError) -> str:
    """What an error response says: its JSON message, else its text, else the reason."""
    raw = error.read().decode("utf-8", errors="replace").strip()
    try:
        body = json.loads(raw)
    except ValueError:
        return raw[:300] or str(error.reason)
    if isinstance(body, dict):
        detail = _error_text(body.get("error")) or body.get("detail") or body.get("message")
        if detail:
            return str(detail)[:300]
    return raw[:300]


# The Chat Completions names for the token counts, and the names that pricing.py reads.
_USAGE_FIELDS = {"prompt_tokens": "input_tokens", "completion_tokens": "output_tokens"}


def token_usage(usage: Any) -> dict[str, int]:
    """The integer token counts of a usage object. Reasoning tokens are output tokens."""
    if not isinstance(usage, dict):
        return {}
    counts = {ours: usage.get(theirs) for theirs, ours in _USAGE_FIELDS.items()}
    return {name: value for name, value in counts.items() if isinstance(value, int)}
