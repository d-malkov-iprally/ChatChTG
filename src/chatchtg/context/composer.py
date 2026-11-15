"""The context composer interface.

A composer turns one interaction into a Context. It is pure: no file, network
or LLM access. Everything it needs comes in through ComposerInput.
"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel

from chatchtg.context.types import Context
from chatchtg.documents.metadata import DocumentRecord


class ComposerInput(BaseModel):
    document: DocumentRecord
    content: bytes
    message: str


class Composer(Protocol):
    name: str
    version: int
    """Increase when the composed context changes, so results stay comparable."""

    def compose(self, request: ComposerInput) -> Context: ...


# A composer may delegate: it then has ``parts`` (the composers it hands requests to) and
# ``delegate(request)`` (the part that builds the context of that request), and its
# ``compose`` returns that part's context unchanged. See ``context/combined.py``.


def composer_id(composer: Composer) -> str:
    """``name@version``. A delegating composer adds the ids of its parts:
    ``combined@1+areas@2+room-conventions@3``, so its id changes when a part changes."""
    own = f"{composer.name}@{composer.version}"
    parts = getattr(composer, "parts", ())
    return "+".join([own, *(composer_id(part) for part in parts)])


def context_composer(composer: Composer, request: ComposerInput) -> Composer | None:
    """The composer that built the context of ``request``, when ``composer`` delegates.

    None for a composer that builds its contexts itself.
    """
    delegate = getattr(composer, "delegate", None)
    if delegate is None:
        return None
    chosen: Composer = delegate(request)
    return context_composer(chosen, request) or chosen
