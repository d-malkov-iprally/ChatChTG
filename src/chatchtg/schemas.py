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
    """How code made the answer (a room count or an area total) when the LLM did not: ``labels``,
    ``shapes``, ``usual``, ``printed``, ``measured``, ... The words are in ``chatchtg.evidence``."""
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    """Where the answer comes from on the drawing: items as the contract in ``chatchtg.evidence``
    says, for example ``{"text", "box"}`` with the box in image pixels (x0, y0, x1, y1)."""
