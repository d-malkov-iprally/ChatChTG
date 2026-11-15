"""The context composer interface.

A composer turns one interaction into a Context. It is pure: no file, network
or LLM access. Everything it needs comes in through ComposerInput.
"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, Field

from chatchtg.context.types import Context
from chatchtg.documents.metadata import DocumentRecord
from chatchtg.extractions.models import Extraction


class ComposerInput(BaseModel):
    document: DocumentRecord
    content: bytes
    message: str
    extractions: list[Extraction] = Field(default_factory=list)
    """The stored extractions of the drawing that the composer reads (``composer_reads``),
    those that exist. The answer service loads them, so the composer reads no file."""


class Composer(Protocol):
    name: str
    version: int
    """Increase when the composed context changes, so results stay comparable."""

    def compose(self, request: ComposerInput) -> Context: ...


# A composer may read stored extractions of the drawing (``extractions/``): it then has
# ``reads``, a tuple of ``(producer, version)``, and finds them in ``ComposerInput.extractions``.
#
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


def composer_reads(composer: Composer) -> tuple[tuple[str, int], ...]:
    """The ``(producer, version)`` of every stored extraction that ``composer`` or one of its
    parts reads, without repeats."""
    found: list[tuple[str, int]] = list(getattr(composer, "reads", ()))
    for part in getattr(composer, "parts", ()):
        found.extend(composer_reads(part))
    return tuple(dict.fromkeys(found))
