"""Area composer: answers area questions on floor plans from a measured room table.

For a question that asks for an area, the pipeline in ``chatchtg.areas`` reads the room
labels, measures the rooms and adds up the ones the question asks for. When the pipeline
is confident, its result is stated in code: the context carries the finished answer text
(``Context.computed.statement``) and the answer service asks no LLM. When the pipeline
cannot answer (a room is missing, the question is not one it understands), the LLM gets
the image as in the baseline, with the room table as extra evidence.

When the pipeline's result is the answer, the context also carries the rooms that were
added up (``method``, ``evidence``), so that the app can draw them on the drawing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

from chatchtg.areas.pipeline import PlanAnalysis, PlanRoom, analyse_bytes
from chatchtg.areas.questions import AreaAnswer, AreaQuery, answer_query, parse_question
from chatchtg.context.baseline import SYSTEM_PROMPT, BaselineComposer
from chatchtg.context.composer import ComposerInput
from chatchtg.context.types import ComputedAnswer, Context, ImagePart, TextPart

AREA_SYSTEM_PROMPT = (
    "You answer questions about one construction drawing. "
    "A measurement tool has already read the drawing: it found the room labels, measured "
    "the rooms and added up the rooms the question asks for. "
    "Give the tool's result as the answer, in one short sentence, and end with the number "
    "and its unit. Do not round the number and do not recompute it."
)

RASTER_TYPES = frozenset({"image/png", "image/jpeg"})

SCALE_DOUBT = "scale-1:200"
"""The plan's scale is 1:200 or coarser, so a room is small on the sheet."""
GAP_DOUBT = "numbering-gap"
"""The room numbering has a gap: a room's label was probably not read."""

_OBTAINED = {"printed": "printed on the plan", "measured": "measured from the drawn lines"}

logger = logging.getLogger(__name__)


def _format_area(value: float) -> str:
    return f"{value:.2f}"


def _scale_text(analysis: PlanAnalysis) -> str:
    if analysis.scale_read:
        return f"scale 1:{analysis.scale}"
    return f"scale not read, 1:{analysis.scale} assumed, so the measured areas may be wrong"


def result_text(analysis: PlanAnalysis, decision: AreaDecision) -> str:
    answer = decision.answer
    lines = [f"Scale of the drawing: 1:{analysis.scale}.", answer.note, "Rooms added up:"]
    for room, method in zip(answer.rooms, decision.methods):
        name = room.name or "no name"
        if room.area is None:  # only a room that was measured together with another one
            lines.append(f"- {room.number} {name}: included in the area of {room.region}")
            continue
        lines.append(f"- {room.number} {name}: {_format_area(room.area)} m² ({method})")
    lines.append(f"Result: {_format_area(answer.value)} m²")
    return "\n".join(lines)


def table_text(analysis: PlanAnalysis) -> str:
    lines = [
        f"A measurement tool read this drawing ({_scale_text(analysis)}). It could not answer "
        "the question alone. Its room table, which may be incomplete:"
    ]
    for room in analysis.rooms:
        area = f"{_format_area(room.area)} m² ({room.method})" if room.area is not None else "area unknown"
        if room.region is not None and room.region != room.number:
            area = f"measured together with {room.region}"
        elif room.region is not None:
            area += ", this is the area of the room together with its neighbours listed below"
        where = f", apartment {room.apartment}" if room.apartment else ""
        outdoor = ", outdoor" if room.outdoor else ""
        lines.append(f"- {room.number} {room.name or 'no name'}: {area}{where}{outdoor}")
    if analysis.missing_numbers:
        lines.append("Room numbers the tool did not find: " + ", ".join(analysis.missing_numbers) + ".")
    return "\n".join(lines)


@dataclass(frozen=True)
class AreaDecision:
    """What the pipeline found for one question, and whether its result is the answer."""

    answer: AreaAnswer
    confident: bool
    """True: the pipeline's result is the answer. False: the LLM gets the image."""
    basis: Literal["printed", "measured"] | None
    """"measured" when any counted room was measured. None when there is no value."""
    counted: tuple[PlanRoom, ...]
    """The rooms of the plan that were added up, in the order of ``answer.rooms``."""
    methods: tuple[str | None, ...]
    """How the area of each counted room was obtained, in the order of ``answer.rooms``.
    It is kept for each reading and not for each room number: a number that was read twice
    can have a printed and a measured reading, and a total counts both."""
    doubts: tuple[str, ...]
    """``SCALE_DOUBT`` and ``GAP_DOUBT``, for every kind of question. They do not change
    ``confident``."""


def decide(query: AreaQuery, analysis: PlanAnalysis) -> AreaDecision:
    rows = analysis.rows()
    # ``answer_query`` returns the rows it was given, so a row finds its reading.
    room_of = {id(row): room for row, room in zip(rows, analysis.rooms)}
    answer = answer_query(query, rows)
    counted = tuple(room_of[id(row)] for row in answer.rooms)
    methods = tuple(room.method for room in counted)
    measured = any(method != "printed" for method in methods)
    # A total may lack a room whose label was not read. On a plan that prints its areas
    # the LLM reads the labels reliably, so it gets the image then. On a plan without
    # printed areas it would have to estimate, and the tool's total is the better answer
    # (measured: right 3 times of 4 even with a gap in the numbering).
    doubtful = analysis.may_be_incomplete(query.kind) and analysis.prints_areas()
    if not analysis.scale_read:
        # A measured area depends on the scale. With a guessed scale it can be 4 times
        # off, so the LLM gets the image. A printed area does not depend on the scale.
        doubtful = doubtful or measured
    doubts = []
    if analysis.scale >= 200:
        doubts.append(SCALE_DOUBT)
    if analysis.missing_numbers:
        doubts.append(GAP_DOUBT)
    basis = None if answer.value is None else "measured" if measured else "printed"
    return AreaDecision(
        answer=answer,
        confident=answer.value is not None and not doubtful,
        basis=basis,
        counted=counted,
        methods=methods,
        doubts=tuple(doubts),
    )


def statement_text(query: AreaQuery, analysis: PlanAnalysis, decision: AreaDecision) -> str:
    """The answer to a question that the pipeline answers with confidence, as Markdown.

    It says what was added up, how each area was obtained and what the pipeline doubts.
    Its last line is the total: the eval rule reads the last number of an answer, so no
    number may follow it.
    """
    answer = decision.answer
    how = "This result was read from the drawing by the measurement tool, without a language model."
    if analysis.scale_read:  # a guessed scale is not a fact about the drawing
        how += f" Scale of the drawing: 1:{analysis.scale}."
    rooms = []
    for room, method in zip(answer.rooms, decision.methods):
        name = room.name or "no name"
        if room.area is None:  # only a room that was measured together with another one
            rooms.append(f"- {room.number} {name}: included in the area of {room.region}")
            continue
        obtained = _OBTAINED.get(method or "", "method unknown")
        rooms.append(f"- {room.number} {name}: {_format_area(room.area)} m² ({obtained})")
    cautions = []
    if decision.basis == "measured":
        cautions.append("A measured area can differ from the true area by a few percent.")
        if SCALE_DOUBT in decision.doubts:
            cautions.append(
                "Caution: at the scale of this drawing a room is small on the sheet, and "
                "measured areas are less reliable."
            )
    if GAP_DOUBT in decision.doubts and analysis.may_be_incomplete(query.kind):
        cautions.append(
            f"Caution: the room numbering has a gap ({', '.join(analysis.missing_numbers)}), "
            "so a room may be missing from this total."
        )
    paragraphs = [answer.note, how, "\n".join(rooms)]
    if cautions:
        paragraphs.append(" ".join(cautions))
    paragraphs.append(f"Answer: {_format_area(answer.value)} m²")
    return "\n\n".join(paragraphs)


def area_method(counted: list[PlanRoom]) -> str:
    """The method of an area answer: ``printed``, ``measured`` or ``printed+measured``."""
    methods = {room.method for room in counted if room.area is not None}
    return "+".join(method for method in ("printed", "measured") if method in methods)


def _place(room: PlanRoom) -> dict[str, Any]:
    """The outline of a room, or the place of its label when no outline was found."""
    if room.polygon:
        return {"polygon": [[round(x), round(y)] for x, y in room.polygon]}
    return {"point": [round(room.seed[0]), round(room.seed[1])]}


def area_evidence(analysis: PlanAnalysis, counted: list[PlanRoom], answer: AreaAnswer) -> list[dict[str, Any]]:
    """The rooms that were added up, as evidence items (see ``chatchtg.evidence``).

    One item for each room, one item for rooms that were measured as one region, and the
    total last. Coordinates are whole pixels of the image.
    """
    items: list[dict[str, Any]] = []
    regions_done: set[str] = set()
    for room in counted:
        if room.region is None:
            items.append(
                {
                    "text": room.number,
                    "detail": room.name or "no name",
                    "kind": room.method,
                    "value": round(room.area, 2),
                    "unit": "m²",
                    **_place(room),
                }
            )
            continue
        if room.region in regions_done:
            continue
        regions_done.add(room.region)
        members = [member for member in counted if member.region == room.region]
        # The room that carries the area and the outline of the whole region.
        carrier = next(
            (r for r in [*members, *analysis.rooms] if r.number == room.region and r.area is not None), room
        )
        items.append(
            {
                "text": "+".join(member.number for member in members),
                "detail": ", ".join(member.name or "no name" for member in members)
                + ": measured as one region, no wall was found between them",
                "kind": "shared",
                "value": round(carrier.area, 2) if carrier.area is not None else None,
                "unit": "m²",
                **_place(carrier),
            }
        )
    items.append(
        {"text": "Total", "detail": answer.note, "kind": "total", "value": round(answer.value, 2), "unit": "m²"}
    )
    return items


class AreaComposer:
    name = "areas"
    version = 3

    def __init__(self) -> None:
        self._baseline = BaselineComposer()

    def compose(self, request: ComposerInput) -> Context:
        query = parse_question(request.message)
        if query is None or request.document.media_type not in RASTER_TYPES:
            return self._baseline.compose(request)
        try:
            analysis = analyse_bytes(request.content)
        except Exception:  # an image the pipeline cannot read is answered as in the baseline
            # Logged, because a fault in the pipeline would otherwise look like a baseline run.
            logger.warning("The area pipeline failed on %s", request.document.filename, exc_info=True)
            return self._baseline.compose(request)
        decision = decide(query, analysis)
        if decision.confident:
            # The statement is the answer. The system prompt and the parts are sent to no
            # LLM; they keep the eval row readable.
            return Context(
                system=AREA_SYSTEM_PROMPT,
                parts=[
                    TextPart(text=result_text(analysis, decision)),
                    TextPart(text=request.message),
                ],
                computed=ComputedAnswer(
                    value=decision.answer.value,
                    basis=decision.basis,
                    doubts=list(decision.doubts),
                    statement=statement_text(query, analysis, decision),
                ),
                method=area_method(list(decision.counted)),
                evidence=area_evidence(analysis, list(decision.counted), decision.answer),
            )
        return Context(
            system=SYSTEM_PROMPT,
            parts=[
                ImagePart(media_type=request.document.media_type, data=request.content),
                TextPart(text=table_text(analysis)),
                TextPart(text=request.message),
            ],
        )
