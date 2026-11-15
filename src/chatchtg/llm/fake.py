"""A deterministic stand-in for a real LLM, for tests and offline runs."""

from __future__ import annotations

from chatchtg.context.types import Context, ImagePart, TextPart
from chatchtg.llm.client import Completion, Location


class FakeLLM:
    model = "fake"

    def __init__(self, reply: str | None = None, locations: list[Location] | None = None) -> None:
        """``locations`` are returned when the context asks for locations."""
        self._reply = reply
        self._locations = locations

    def complete(self, context: Context) -> Completion:
        has_image = any(isinstance(part, ImagePart) for part in context.parts)
        texts = [part.text for part in context.parts if isinstance(part, TextPart)]
        text = self._reply
        if text is None:
            # No digits of our own here: the eval scorer reads numbers from the answer.
            seen = "a drawing" if has_image else "no drawing"
            text = f"[fake answer] I received {seen} and the message: {' '.join(texts)}"
        locations = list(self._locations or []) if context.locations is not None else None
        return Completion(text=text, model=self.model, locations=locations)
