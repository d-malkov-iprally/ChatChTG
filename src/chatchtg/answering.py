"""The answer service: (document, message) -> response.

The user interface and the eval runner both call this service.
"""

from __future__ import annotations

from collections import Counter
from typing import Protocol

from pydantic import BaseModel

from chatchtg.context.composer import Composer, ComposerInput, composer_id
from chatchtg.context.types import Context
from chatchtg.documents.library import DocumentLibrary
from chatchtg.llm.client import Completion, LLMClient
from chatchtg.objects.counter import ObjectFinding
from chatchtg.rooms.router import Finding
from chatchtg.schemas import AnswerRequest, AnswerResponse

ROOM_COUNTER_MODEL = "room-counter"
"""The model name of an answer that code made without an LLM call."""
OBJECT_COUNTER_MODEL = "object-counter"
"""The model name of an answer that code made from the stored detections of the drawing."""


class RoomCounter(Protocol):
    def solve(self, content: bytes, message: str) -> Finding | None:
        """Count rooms for a room-count question. None when it is another question."""
        ...


class ObjectCounter(Protocol):
    def solve(self, document_id: str, message: str) -> ObjectFinding | None:
        """Count objects for an object-count question. None when it is another question, or
        when the drawing has no stored detections."""
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
        object_counter: ObjectCounter | None = None,
    ) -> None:
        """With a ``room_counter``, a room-count question is answered in code when the counter
        is confident, and by the LLM otherwise. ``trust_room_counter`` uses the counter's answer
        even when it is not confident. With an ``object_counter``, an object-count question is
        answered from the stored detections of the drawing when it has them.
        """
        self._library = library
        self._composer = composer
        self._llm = llm
        self._room_counter = room_counter
        self._trust_room_counter = trust_room_counter
        self._object_counter = object_counter

    def answer(self, request: AnswerRequest) -> AnswerResponse:
        return self.answer_traced(request).response

    def answer_traced(self, request: AnswerRequest) -> AnswerTrace:
        document = self._library.get(request.document_id)
        content = self._library.content(document)
        context = self._composer.compose(
            ComposerInput(document=document, content=content, message=request.message)
        )
        finding: Finding | ObjectFinding | None = None
        if self._object_counter is not None:
            finding = self._object_counter.solve(document.id, request.message)
        if finding is None and self._room_counter is not None:
            finding = self._room_counter.solve(content, request.message)
            if finding is not None and not (finding.confident or self._trust_room_counter):
                finding = None
        if isinstance(finding, ObjectFinding):
            completion = Completion(text=_object_text(finding), model=OBJECT_COUNTER_MODEL)
        elif finding is not None:
            completion = Completion(text=_finding_text(finding), model=ROOM_COUNTER_MODEL)
        else:
            completion = self._llm.complete(context)
        response = AnswerResponse(
            text=completion.text,
            composer=composer_id(self._composer),
            model=completion.model,
            method=finding.method if finding else None,
            evidence=finding.evidence if finding else [],
        )
        return AnswerTrace(response=response, context=context, completion=completion)


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
_METHODS = {
    "labels": "counted from the labels printed on the drawing",
    "shapes": "counted from the shapes on the drawing",
    "labels+shapes": "counted from the printed labels and confirmed by the shapes",
    "usual": "not counted on the drawing: this is the usual number for such a plan",
}


def _finding_text(finding: Finding) -> str:
    """The answer for a count made in code. It ends with the number, which the eval rule reads."""
    question = finding.question
    subject = f"rooms labelled '{question.label}'" if question.kind == "label" else _SUBJECTS[question.kind]
    read = ", ".join(item["text"] for item in finding.evidence[:12])
    seen = f" Seen: {read}." if read and finding.method.startswith("labels") else ""
    return f"Number of {subject}, {_METHODS[finding.method]}.{seen} Answer: {finding.count}"


_OBJECTS = {
    "window": "windows",
    "door": "doors",
    "toilet": "toilet bowls",
    "sink": "sinks and wash basins",
    "bath_shower": "showers and bathtubs",
}


def _object_text(finding: ObjectFinding) -> str:
    """The answer for an object count made from detections. It ends with the number."""
    found = Counter(item["text"].rsplit(" ", 1)[0] for item in finding.evidence)
    parts = ", ".join(f"{n} {label}" for label, n in sorted(found.items()))
    seen = f" Found: {parts}." if len(found) > 1 else ""
    return (
        f"Number of {_OBJECTS[finding.question.kind]}, counted from the symbols that the "
        f"detector found on the drawing (each one is marked with its box).{seen} "
        f"Answer: {finding.count}"
    )
