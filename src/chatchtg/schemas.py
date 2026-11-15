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
    """Name and version of the composer that built the context, e.g. ``baseline@1``. A
    composer that delegates names its parts too: ``combined@1+areas@2+room-conventions@3``."""
    context_composer: str | None = None
    """When the composer delegates: name and version of the part that built this context,
    e.g. ``areas@2``. None for a composer that builds its contexts itself."""
    model: str
    """Model name reported by the LLM client."""
    method: str | None = None
    """How the answer was made when code made it (a room count or an area total: ``labels``,
    ``shapes``, ``usual``, ``printed``, ``measured``, ...) or when the LLM marked what it counted
    (``llm-located``). The words are in ``chatchtg.evidence``."""
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    """Where the answer comes from on the drawing: items as the contract in ``chatchtg.evidence``
    says, for example ``{"text", "box"}`` with the box in image pixels (x0, y0, x1, y1), or
    ``{"text", "kind": "llm", "point"}`` for a position that the LLM gave."""
    evidence_note: str | None = None
    """What a check of the evidence found, for example a count that differs from the number of
    marks. None when nothing was found."""
