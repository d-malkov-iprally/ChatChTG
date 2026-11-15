"""The answer service: (document, message) -> response.

The user interface and the eval runner both call this service.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

from pydantic import BaseModel

from chatchtg.context.composer import Composer, ComposerInput, composer_id
from chatchtg.context.types import Context, ImagePart
from chatchtg.documents.library import DocumentLibrary
from chatchtg.evals.scoring import extract_number
from chatchtg.evidence import METHOD_WORDS
from chatchtg.evidence_check import LLM_LOCATED, CheckedEvidence, check_locations
from chatchtg.llm.client import Completion, LLMClient
from chatchtg.rooms.router import Finding
from chatchtg.schemas import AnswerRequest, AnswerResponse

logger = logging.getLogger(__name__)

ROOM_COUNTER_MODEL = "room-counter"
"""The model name of an answer that code made without an LLM call."""


class RoomCounter(Protocol):
    def solve(self, content: bytes, message: str) -> Finding | None:
        """Count rooms for a room-count question. None when it is another question."""
        ...


class AnswerTrace(BaseModel):
    """A response together with what produced it. For evals and debugging."""

    response: AnswerResponse
    context: Context
    completion: Completion


class AnswerService:
    def __init__(
        self,
        library: DocumentLibrary,
        composer: Composer,
        llm: LLMClient,
        room_counter: RoomCounter | None = None,
        trust_room_counter: bool = False,
    ) -> None:
        """With a ``room_counter``, a room-count question is answered in code when the counter
        is confident, and by the LLM otherwise. ``trust_room_counter`` uses the counter's answer
        even when it is not confident.
        """
        self._library = library
        self._composer = composer
        self._llm = llm
        self._room_counter = room_counter
        self._trust_room_counter = trust_room_counter

    def answer(self, request: AnswerRequest) -> AnswerResponse:
        return self.answer_traced(request).response

    def answer_traced(self, request: AnswerRequest) -> AnswerTrace:
        document = self._library.get(request.document_id)
        content = self._library.content(document)
        context = self._composer.compose(
            ComposerInput(document=document, content=content, message=request.message)
        )
        finding = None
        if self._room_counter is not None:
            finding = self._room_counter.solve(content, request.message)
        if finding is not None and (finding.confident or self._trust_room_counter):
            completion = Completion(text=_finding_text(finding), model=ROOM_COUNTER_MODEL)
        else:
            finding = None
            completion = self._llm.complete(context)
        # The evidence of one producer, in this order: the room counter when its count is the
        # answer; what the composer knows about a result that the context states; the positions
        # that the LLM gave for the items it counted; none.
        method: str | None = None
        evidence: list[dict[str, Any]] = []
        note: str | None = None
        if finding is not None:
            method, evidence = finding.method, finding.evidence
        elif context.method is not None or context.evidence:
            method, evidence = context.method, context.evidence
            if completion.locations is not None:
                logger.warning(
                    "The context states a code-made answer, so the positions of the LLM were "
                    "not used. A composer asks for locations only where the LLM decides the answer."
                )
        else:
            checked = _llm_evidence(context, completion)
            if checked is not None:
                method, evidence, note = LLM_LOCATED, checked.evidence, checked.note
        response = AnswerResponse(
            text=completion.text,
            composer=composer_id(self._composer),
            model=completion.model,
            method=method,
            evidence=evidence,
            evidence_note=note,
        )
        return AnswerTrace(response=response, context=context, completion=completion)


def _llm_evidence(context: Context, completion: Completion) -> CheckedEvidence | None:
    """The checked positions of a completion that has locations. None when it has none.

    The stated count is read from the text exactly as the eval rule reads it.
    """
    if completion.locations is None:
        return None
    image = next((part for part in context.parts if isinstance(part, ImagePart)), None)
    size = image.size() if image is not None else None
    return check_locations(completion.locations, size, extract_number(completion.text))


_SUBJECTS = {
    "bedroom": "bedrooms",
    "sauna": "saunas",
    "kitchen": "kitchens",
    "bathroom": "bathrooms or washrooms",
    "wc": "separate WC rooms",
    "balcony": "balconies",
    "storeys": "floors (storeys)",
    "apartments": "apartments",
}


def _finding_text(finding: Finding) -> str:
    """The answer for a count made in code. It ends with the number, which the eval rule reads."""
    question = finding.question
    subject = f"rooms labelled '{question.label}'" if question.kind == "label" else _SUBJECTS[question.kind]
    read = ", ".join(item["text"] for item in finding.evidence[:12])
    seen = f" Seen: {read}." if read and finding.method.startswith("labels") else ""
    return f"Number of {subject}, {METHOD_WORDS[finding.method]}.{seen} Answer: {finding.count}"
