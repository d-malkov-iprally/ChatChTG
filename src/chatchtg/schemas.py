"""The user-facing interaction contract: (document, message) -> response.

Keep every request/response field here. New fields must be optional or have
defaults, so that existing callers keep working.
"""

from pydantic import BaseModel


class AnswerRequest(BaseModel):
    document_id: str
    message: str


class AnswerResponse(BaseModel):
    text: str
    composer: str
    """Name and version of the composer that built the context, e.g. ``baseline@1``."""
    model: str
    """Model name reported by the LLM client."""
