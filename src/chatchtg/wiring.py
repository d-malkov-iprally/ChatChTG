"""Builds the concrete objects from settings. Entry points (UI, CLI) call this."""

from __future__ import annotations

import logging

from chatchtg.answering import AnswerService
from chatchtg.config import Settings
from chatchtg.context.registry import get_composer
from chatchtg.documents.blobs import LocalBlobStore
from chatchtg.documents.library import DocumentLibrary
from chatchtg.documents.metadata import SqliteMetadataStore
from chatchtg.evals.judge import DEFAULT_JUDGE_MODEL, ClaudeJudge, FakeJudge, Judge
from chatchtg.extractions.store import LocalExtractionStore
from chatchtg.llm.client import build_llm
from chatchtg.rooms import ocr
from chatchtg.rooms.counter import LabelRoomCounter

ROOM_COUNTER_MODES = ("off", "confident", "always")

logger = logging.getLogger(__name__)


def build_library(settings: Settings) -> DocumentLibrary:
    return DocumentLibrary(
        blobs=LocalBlobStore(settings.state_dir / "blobs", settings.data_root),
        metadata=SqliteMetadataStore(settings.state_dir / "chatchtg.db"),
    )


def build_extraction_store(settings: Settings) -> LocalExtractionStore:
    return LocalExtractionStore(settings.state_dir / "metadata")


def build_answer_service(
    settings: Settings, library: DocumentLibrary, composer: str | None = None
) -> AnswerService:
    if settings.room_counter not in ROOM_COUNTER_MODES:
        raise ValueError(
            f"Unknown room counter mode {settings.room_counter!r}. "
            f"Available: {', '.join(ROOM_COUNTER_MODES)}"
        )
    room_counter = None
    if settings.room_counter != "off":
        if not ocr.available():
            logger.warning(
                "CHATCHTG_ROOM_COUNTER=%s has no effect on this machine: the room counter needs "
                "Apple Vision (macOS). The LLM answers every question.",
                settings.room_counter,
            )
        room_counter = LabelRoomCounter(settings.state_dir / "ocr-cache")
    return AnswerService(
        library=library,
        composer=get_composer(composer or settings.composer),
        llm=build_llm(settings.llm_provider, settings.llm_model, settings.anthropic_api_key),
        room_counter=room_counter,
        trust_room_counter=settings.room_counter == "always",
    )


def build_judge(settings: Settings) -> Judge:
    """Return the eval judge. With the fake LLM provider the judge is fake too."""
    if settings.llm_provider == "fake":
        return FakeJudge()
    return ClaudeJudge(settings.judge_model or DEFAULT_JUDGE_MODEL, settings.anthropic_api_key)
