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


def composer_id(composer: Composer) -> str:
    return f"{composer.name}@{composer.version}"
