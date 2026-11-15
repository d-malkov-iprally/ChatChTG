"""The answer service: (document, message) -> response.

The user interface and the eval runner both call this service.
"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel

from chatchtg.context.composer import Composer, ComposerInput, composer_id
from chatchtg.context.types import Context
from chatchtg.documents.library import DocumentLibrary
from chatchtg.evidence import METHOD_WORDS
from chatchtg.llm.client import Completion, LLMClient
from chatchtg.rooms.router import Finding
from chatchtg.schemas import AnswerRequest, AnswerResponse

ROOM_COUNTER_MODEL = "room-counter"
"""The model name of an answer that code made without an LLM call."""
AREA_PIPELINE_MODEL = "area-pipeline"
"""The model name of an area answer that code computed and stated without an LLM call."""


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
            completion = self._state(context)
        response = AnswerResponse(
            text=completion.text,
            composer=composer_id(self._composer),
            model=completion.model,
            # The evidence of one producer: the room counter when its count is the answer,
            # otherwise what the composer knows about a result that the context states.
            method=finding.method if finding else context.method,
            evidence=finding.evidence if finding else context.evidence,
        )
        return AnswerTrace(response=response, context=context, completion=completion)

    def _state(self, context: Context) -> Completion:
        """The answer text for a context: the statement that code wrote, when the context
        carries one, and the LLM's completion otherwise.

        Only the areas composer writes a statement today, so the model name is the area
        pipeline's. A second composer that states answers needs its own name.
        """
        computed = context.computed
        if computed is not None and computed.statement:
            return Completion(text=computed.statement, model=AREA_PIPELINE_MODEL)
        return self._llm.complete(context)


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
