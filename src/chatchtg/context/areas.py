"""Area composer: answers area questions on floor plans from a measured room table.

For a question that asks for an area, the pipeline in ``chatchtg.areas`` reads the room
labels, measures the rooms and adds up the ones the question asks for. The LLM gets that
result as text and states the answer; it does not get the image, which is 98% of the
input tokens of the baseline. When the pipeline cannot answer (a room is missing, the
question is not one it understands), the LLM gets the image as in the baseline, with the
room table as extra evidence.
"""

from __future__ import annotations

from chatchtg.areas.pipeline import PlanAnalysis, analyse_bytes
from chatchtg.areas.questions import AreaAnswer, parse_question, answer_query
from chatchtg.context.baseline import SYSTEM_PROMPT, BaselineComposer
from chatchtg.context.composer import ComposerInput
from chatchtg.context.types import Context, ImagePart, TextPart

AREA_SYSTEM_PROMPT = (
    "You answer questions about one construction drawing. "
    "A measurement tool has already read the drawing: it found the room labels, measured "
    "the rooms and added up the rooms the question asks for. "
    "Give the tool's result as the answer, in one short sentence, and end with the number "
    "and its unit. Do not round the number and do not recompute it."
)

RASTER_TYPES = frozenset({"image/png", "image/jpeg"})


def _format_area(value: float) -> str:
    return f"{value:.2f}"


def result_text(analysis: PlanAnalysis, answer: AreaAnswer) -> str:
    lines = [f"Scale of the drawing: 1:{analysis.scale}.", answer.note, "Rooms added up:"]
    methods = {room.number: room.method for room in analysis.rooms}
    for room in answer.rooms:
        name = room.name or "no name"
        lines.append(f"- {room.number} {name}: {_format_area(room.area)} m² ({methods.get(room.number)})")
    lines.append(f"Result: {_format_area(answer.value)} m²")
    return "\n".join(lines)


def table_text(analysis: PlanAnalysis) -> str:
    lines = [
        f"A measurement tool read this drawing (scale 1:{analysis.scale}). It could not answer "
        "the question alone. Its room table, which may be incomplete:"
    ]
    for room in analysis.rooms:
        area = f"{_format_area(room.area)} m² ({room.method})" if room.area is not None else "area unknown"
        where = f", apartment {room.apartment}" if room.apartment else ""
        outdoor = ", outdoor" if room.outdoor else ""
        lines.append(f"- {room.number} {room.name or 'no name'}: {area}{where}{outdoor}")
    if analysis.missing_numbers:
        lines.append("Room numbers the tool did not find: " + ", ".join(analysis.missing_numbers) + ".")
    return "\n".join(lines)


class AreaComposer:
    name = "areas"
    version = 1

    def __init__(self) -> None:
        self._baseline = BaselineComposer()

    def compose(self, request: ComposerInput) -> Context:
        query = parse_question(request.message)
        if query is None or request.document.media_type not in RASTER_TYPES:
            return self._baseline.compose(request)
        try:
            analysis = analyse_bytes(request.content)
        except Exception:  # an image the pipeline cannot read is answered as in the baseline
            return self._baseline.compose(request)
        answer = answer_query(query, analysis.rows())
        # A total may lack a room whose label was not read. On a plan that prints its areas
        # the LLM reads the labels reliably, so it gets the image then. On a plan without
        # printed areas it would have to estimate, and the tool's total is the better answer
        # (measured: right 3 times of 4 even with a gap in the numbering).
        doubtful = analysis.may_be_incomplete(query.kind) and analysis.prints_areas()
        if answer.value is not None and not doubtful:
            return Context(
                system=AREA_SYSTEM_PROMPT,
                parts=[TextPart(text=result_text(analysis, answer)), TextPart(text=request.message)],
            )
        return Context(
            system=SYSTEM_PROMPT,
            parts=[
                ImagePart(media_type=request.document.media_type, data=request.content),
                TextPart(text=table_text(analysis)),
                TextPart(text=request.message),
            ],
        )
