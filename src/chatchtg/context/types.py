"""The context: everything the LLM is given, in a provider-neutral form.

It also carries what the composer knows about the origin of a code-made answer
(``method``, ``evidence``). The LLM adapters do not read those two fields. They do read
``locations``: a request to the model for the position of each item it counted.
"""

from __future__ import annotations

import hashlib
import io
from typing import Any, Literal

from pydantic import BaseModel, Field


class TextPart(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ImagePart(BaseModel):
    type: Literal["image"] = "image"
    media_type: str
    data: bytes

    def size(self) -> tuple[int, int] | None:
        """``(width, height)`` of the stored pixels, without EXIF rotation. None when the
        bytes are not a readable image."""
        from PIL import Image

        try:
            with Image.open(io.BytesIO(self.data)) as image:
                return image.size
        except Exception:  # not an image, or one that Pillow cannot read
            return None


class LocationRequest(BaseModel):
    """Asks the model to give the position of each item it counted, next to its answer."""

    shape: Literal["point"] = "point"


class Context(BaseModel):
    system: str
    parts: list[TextPart | ImagePart]
    method: str | None = None
    """How code made the answer that this context states. Not sent to the LLM."""
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    """Where that answer comes from on the drawing (see chatchtg.evidence). Not sent to the LLM."""
    locations: LocationRequest | None = None
    """Read by the LLM adapters. None: a plain text reply. Set: the adapter also asks for the
    position of each counted item and returns them in ``Completion.locations``. An adapter
    that cannot do that ignores the request. Set it only where the LLM decides the answer."""

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
        summary: dict[str, Any] = {"system": self.system, "parts": parts}
        if self.locations is not None:
            summary["locations"] = self.locations.model_dump()
        return summary
