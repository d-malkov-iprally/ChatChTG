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
from chatchtg.objects.counter import DEFAULT_SKIP, DetectorObjectCounter, parse_skip
from chatchtg.rooms import ocr
from chatchtg.rooms.counter import LabelRoomCounter

ROOM_COUNTER_MODES = ("off", "confident", "always")
OBJECT_COUNTER_MODES = ("off", "detections")

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
    if settings.object_counter not in OBJECT_COUNTER_MODES:
        raise ValueError(
            f"Unknown object counter mode {settings.object_counter!r}. "
            f"Available: {', '.join(OBJECT_COUNTER_MODES)}"
        )
    object_counter = None
    if settings.object_counter == "detections":
        skip = DEFAULT_SKIP if settings.object_counter_skip is None else parse_skip(settings.object_counter_skip)
        object_counter = DetectorObjectCounter(build_extraction_store(settings), skip=skip)
    room_counter = None
    if settings.room_counter != "off":
        if not ocr.available():
            logger.warning(
                "CHATCHTG_ROOM_COUNTER=%s has no effect on this machine: the room counter needs "
                "Apple Vision (macOS). The LLM answers every question.",
                settings.room_counter,
            )
        room_counter = LabelRoomCounter(settings.state_dir / "ocr-cache")
    if settings.llm_provider == "modal":
        api_key = settings.modal_proxy_token
    else:
        api_key = settings.anthropic_api_key
    return AnswerService(
        library=library,
        composer=get_composer(composer or settings.composer),
        llm=build_llm(settings.llm_provider, settings.llm_model, api_key, settings.modal_endpoint),
        room_counter=room_counter,
        trust_room_counter=settings.room_counter == "always",
        object_counter=object_counter,
    )


def build_judge(settings: Settings) -> Judge:
    """Return the eval judge. With the fake LLM provider the judge is fake too.

    Every other provider gets the Claude judge, so `eval --judge` with the modal provider
    also needs an Anthropic key.
    """
    if settings.llm_provider == "fake":
        return FakeJudge()
    return ClaudeJudge(settings.judge_model or DEFAULT_JUDGE_MODEL, settings.anthropic_api_key)
