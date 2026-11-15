"""The user-facing interaction contract: (document, message) -> response.

Keep every request/response field here. New fields must be optional or have
defaults, so that existing callers keep working.
"""

from typing import Any

from pydantic import BaseModel, Field


class AnswerRequest(BaseModel):
    document_id: str
    message: str


class AnswerResponse(BaseModel):
    text: str
    composer: str
    """Name and version of the composer that built the context, e.g. ``baseline@1``."""
    model: str
    """Model name reported by the LLM client."""
    method: str | None = None
    """How a count was made when code made it, not the LLM: ``labels``, ``shapes``, ``usual``
    (rooms), ``detector`` (objects)."""
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    """What the code counted: ``{"text", "box"}``, the box in image pixels (x0, y0, x1, y1).
    An object also has ``ref``, the reference of the detection in the extraction store."""
