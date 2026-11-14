"""The context: everything the LLM is given, in a provider-neutral form.

It also carries what the composer knows about the origin of a code-made answer
(``method``, ``evidence``). The LLM adapters do not read those two fields.
"""

from __future__ import annotations

import hashlib
from typing import Any, Literal

from pydantic import BaseModel, Field


class TextPart(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ImagePart(BaseModel):
    type: Literal["image"] = "image"
    media_type: str
    data: bytes


class Context(BaseModel):
    system: str
    parts: list[TextPart | ImagePart]
    method: str | None = None
    """How code made the answer that this context states. Not sent to the LLM."""
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    """Where that answer comes from on the drawing (see chatchtg.evidence). Not sent to the LLM."""

    def summary(self) -> dict[str, Any]:
        """A JSON-safe description for logs and eval results. Images become their hash."""
        parts: list[dict[str, Any]] = []
        for part in self.parts:
            if isinstance(part, ImagePart):
                parts.append(
                    {
                        "type": "image",
                        "media_type": part.media_type,
                        "sha256": hashlib.sha256(part.data).hexdigest(),
                        "size_bytes": len(part.data),
                    }
                )
            else:
                parts.append({"type": "text", "text": part.text})
        return {"system": self.system, "parts": parts}
