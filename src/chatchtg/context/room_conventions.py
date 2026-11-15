"""Baseline context plus the counting conventions of Finnish floor plans, for room-count questions.

The plain model reads the labels well. Its misses on room counts come from conventions that
the drawing does not state, mostly from counting the rooms of a neighbouring home that is
partly visible on the sheet. For any other question the context is the baseline one.
"""

from chatchtg.context.baseline import SYSTEM_PROMPT
from chatchtg.context.composer import ComposerInput
from chatchtg.context.types import Context, ImagePart, TextPart
from chatchtg.rooms.questions import classify

_ONE_HOME = (
    "The drawing is the plan of one home. Count only the rooms of that home: the one that is "
    "highlighted, drawn completely, or named in the title. Other homes of the building are often "
    "partly visible beside it, cut off at the edge of the sheet or drawn lighter. Do not count "
    "their rooms, even when their labels are readable."
)
_BY_KIND = {
    "bedroom": (
        "Bedrooms are the rooms marked MH or MAKUUHUONE. A study (TH), "
        "an alcove (ALK), a living room (OH) and a walk-in closet (VH) are not bedrooms."
    ),
    "sauna": "A sauna is marked S or SAUNA, or drawn with benches and a stove.",
    "kitchen": (
        "Kitchens are marked K, KK, KT, KEITTIÖ or TUPAK, also inside a combined label such "
        "as OH+K. A utility kitchen (APUK) of the same home counts as one more. TK is an "
        "entrance vestibule, not a kitchen."
    ),
    "bathroom": (
        "A bathroom or washroom is marked KH, KPH, PH, PESUH or SH. A room that holds only "
        "a toilet is a WC room, not a bathroom."
    ),
    "wc": (
        "A separate WC room is a small room of this home that holds only a toilet, usually "
        "marked WC. A toilet inside a bathroom (KH, KPH) is not a separate WC room."
    ),
    "balcony": (
        "A balcony is marked PARVEKE or PARV, also when glazed. A terrace (TERASSI), a porch "
        "(KUISTI), a patio and a yard are not balconies."
    ),
    "storeys": (
        "Count each separate floor drawing on the sheet as one storey. A basement, a loft and "
        "a detached outbuilding that is drawn as its own plan each count as a storey. Two "
        "drawings of the same floor do not."
    ),
}


def conventions(message: str) -> str | None:
    """The conventions for ``message``, or None when it is not a room count on a Finnish plan."""
    question = classify(message)
    if question is None or question.profile != "finnish":
        return None
    if question.kind == "storeys":
        return _BY_KIND["storeys"]
    return f"{_ONE_HOME} {_BY_KIND[question.kind]}"


class RoomConventionsComposer:
    name = "room-conventions"
    version = 3

    def compose(self, request: ComposerInput) -> Context:
        parts: list[TextPart | ImagePart] = [
            ImagePart(media_type=request.document.media_type, data=request.content)
        ]
        notes = conventions(request.message)
        if notes is not None:
            parts.append(TextPart(text=f"Counting conventions for this kind of drawing: {notes}"))
        parts.append(TextPart(text=request.message))
        return Context(system=SYSTEM_PROMPT, parts=parts)
