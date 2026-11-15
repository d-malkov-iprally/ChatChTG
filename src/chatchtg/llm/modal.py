"""The Modal adapter: sends a Context to a Modal Shared Endpoint (Chat Completions API).

For a context that asks for locations (``Context.locations``), the adapter requests a JSON
object through ``response_format`` and reads the answer text and the item positions from it.
What is known about that, for Kimi K3.1 Pro on a Shared Endpoint, on 2026-11-15:

- The Kimi quickstart documents ``response_format`` of type ``json_schema`` with
  ``strict: true`` for the final ``message.content``
  (https://platform.kimi.ai/docs/guide/kimi-k3-1-pro-quickstart). The Modal endpoint accepted
  it, and ``json_object`` too. The keys came back in alphabetical order, not in schema order,
  so the reading does not depend on the order.
- No documentation of a coordinate convention was found, neither for the model nor for the
  endpoint (the Kimi vision guide, https://platform.kimi.ai/docs/guide/use-kimi-vision-model,
  only says that image tokens are "dynamic"). The convention here is empirical: asked for
  absolute pixel coordinates, the model answers in pixels of the file as it was sent, and the
  marks sat on the objects on the plans that were looked at (docs/benchmarks has the record).
  The endpoint reports ``image_tokens`` equal to the number of 28-pixel patches of the whole
  file, up to 3507x2480 pixels, so it does not make the image smaller before the model sees it.
"""

from __future__ import annotations

import base64
import json
import logging
import math
import ssl
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

import certifi

from chatchtg.context.types import Context, ImagePart
from chatchtg.llm.client import Completion, Location

logger = logging.getLogger(__name__)

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
# For a reply to a location request that is not the JSON object asked for. It is not shown
# as it is, because it can contain coordinates, which the eval scorer would read as the answer.
UNREADABLE_TEXT = "[no answer] The reply of the model could not be read."

# The reply to a location request. ``answer`` is the only text that the user sees.
LOCATED_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "description": "Every item that was counted, one entry for each. Empty when there is none.",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string", "description": "What the item is, in one to three words."},
                    "x": {
                        "type": "integer",
                        "description": "Pixel column of the centre of the item, from the left edge of the image.",
                    },
                    "y": {
                        "type": "integer",
                        "description": "Pixel row of the centre of the item, from the top edge of the image.",
                    },
                },
                "required": ["label", "x", "y"],
                "additionalProperties": False,
            },
        },
        "answer": {
            "type": "string",
            "description": "The short answer for the user, without coordinates. It ends with the number and its unit.",
        },
    },
    "required": ["items", "answer"],
    "additionalProperties": False,
}
# Added to the system prompt of a location request, in case the endpoint does not enforce
# the schema. The discovery calls were made with this sentence and the schema together.
LOCATED_FORMAT_PROMPT = (
    'Reply with one JSON object and nothing else: {"items": [{"label": "...", "x": 0, "y": 0}], '
    '"answer": "..."}. The answer has no coordinates.'
)
# The model sees an image as patches of this many pixels on each side.
PATCH_PIXELS = 28


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
        located = context.locations is not None
        system = f"{context.system} {LOCATED_FORMAT_PROMPT}" if located else context.system
        body: dict[str, Any] = {
            "model": self.endpoint,
            "max_tokens": MAX_TOKENS,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": [_block(part) for part in context.parts]},
            ],
        }
        if located:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "located_answer", "strict": True, "schema": LOCATED_SCHEMA},
            }
        response = self._post(body)
        error = response.get("error")
        if error or not response.get("choices"):
            raise ModalError(f"Modal sent no completion: {_error_text(error) or response}")
        choice = response["choices"][0]
        text = (choice.get("message") or {}).get("content") or ""
        if not text and choice.get("finish_reason") == "length":
            return Completion(
                text=NO_BUDGET_TEXT, model=self.model, usage=token_usage(response.get("usage"))
            )
        locations = None
        if located:
            image = next((part for part in context.parts if isinstance(part, ImagePart)), None)
            text, locations = read_located_reply(
                text,
                image.size() if image is not None else None,
                image_tokens(response.get("usage")),
            )
        return Completion(
            text=text,
            model=self.model,
            usage=token_usage(response.get("usage")),
            locations=locations,
        )

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


def image_tokens(usage: Any) -> int | None:
    """The number of image tokens of the prompt, when the endpoint reports it."""
    details = usage.get("prompt_tokens_details") if isinstance(usage, dict) else None
    count = details.get("image_tokens") if isinstance(details, dict) else None
    return count if isinstance(count, int) and not isinstance(count, bool) else None


def full_size_tokens(size: tuple[int, int]) -> int:
    """The image tokens of a file of ``size`` that the model sees without a resize."""
    return math.ceil(size[0] / PATCH_PIXELS) * math.ceil(size[1] / PATCH_PIXELS)


def to_stored_pixels(
    x: float, y: float, size: tuple[int, int] | None, tokens: int | None
) -> tuple[float, float] | None:
    """Convert a position that the model gave to pixels of the stored image.

    The model answers in pixels of the image as it sees it. The endpoint gives it the file
    at its own size (see the module docstring), so the position is returned as it is.
    ``size`` is the size of the file and ``tokens`` the image tokens that the endpoint
    reported; either can be None when it is not known. If the endpoint reports fewer tokens
    than the whole file has patches, it made the image smaller. Where the model's pixels
    point after that was never observed, so the result is then None: no mark is better than
    a mark in the wrong place.
    """
    if size is not None and tokens is not None and tokens < full_size_tokens(size):
        return None
    return (x, y)


def _is_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:  # an integer too large for a float
        return False


def _json_object(text: str) -> dict[str, Any] | None:
    """The JSON object of a reply: the whole text, or what is between its outer braces."""
    start, end = text.find("{"), text.rfind("}")
    for candidate in (text, text[start : end + 1] if 0 <= start < end else ""):
        try:
            parsed = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def read_located_reply(
    text: str, size: tuple[int, int] | None, tokens: int | None
) -> tuple[str, list[Location] | None]:
    """The answer text and the locations of a reply to a location request.

    Not a JSON object, or no ``answer`` string in it: ``UNREADABLE_TEXT`` and None.
    ``items`` missing or not a list, or an image that the endpoint resized: the answer and
    None. An item without a string label and numbers for ``x`` and ``y`` is left out.
    """
    reply = _json_object(text)
    answer = reply.get("answer") if reply is not None else None
    if not isinstance(answer, str) or not answer.strip():
        logger.warning("The reply to a location request could not be read: %.200r", text)
        return UNREADABLE_TEXT, None
    items = reply.get("items")
    if not isinstance(items, list):
        logger.warning("The reply to a location request has no list of items.")
        return answer, None
    locations: list[Location] = []
    for item in items:
        if not (
            isinstance(item, dict)
            and isinstance(item.get("label"), str)
            and _is_number(item.get("x"))
            and _is_number(item.get("y"))
        ):
            logger.warning("An item of a location reply was left out: %.200r", item)
            continue
        point = to_stored_pixels(item["x"], item["y"], size, tokens)
        if point is None:
            logger.warning("The endpoint resized the image, so the positions are not used.")
            return answer, None
        locations.append(Location(label=item["label"], point=point))
    return answer, locations
