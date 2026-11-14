"""Builds the concrete objects from settings. Entry points (UI, CLI) call this."""

from __future__ import annotations

from chatchtg.answering import AnswerService
from chatchtg.config import Settings
from chatchtg.context.registry import get_composer
from chatchtg.documents.blobs import LocalBlobStore
from chatchtg.documents.library import DocumentLibrary
from chatchtg.documents.metadata import SqliteMetadataStore
from chatchtg.llm.client import build_llm


def build_library(settings: Settings) -> DocumentLibrary:
    return DocumentLibrary(
        blobs=LocalBlobStore(settings.state_dir / "blobs", settings.data_root),
        metadata=SqliteMetadataStore(settings.state_dir / "chatchtg.db"),
    )


def build_answer_service(
    settings: Settings, library: DocumentLibrary, composer: str | None = None
) -> AnswerService:
    return AnswerService(
        library=library,
        composer=get_composer(composer or settings.composer),
        llm=build_llm(settings.llm_provider, settings.llm_model, settings.anthropic_api_key),
    )
