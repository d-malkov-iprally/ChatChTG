"""Object-hints composer: the LLM counts objects with the detector's candidates as a hint.

For an object-count question about a Finnish plan (``objects/questions``) on a drawing whose
detections are stored (producer ``rfdetr-large-1024@1``), the LLM gets the drawing twice:
as it is, and with the candidate boxes of the asked class(es) drawn on it and numbered. A
text lists the candidates with their scores. The candidates are the detections that the
object counter counts, and the doubtful ones: detections that score below the counter's
minimum but at least half of it. Most of the counter's misses are true objects that score
just under its minimum. The LLM checks each candidate, looks for objects without a mark and
gives the count.

The detections come in through ``ComposerInput.extractions``, which the answer service
loads from the extraction store; the composer reads no file and needs no detector. Every
other question, and a drawing without stored detections, gets the baseline context.
"""

from __future__ import annotations

import io
import logging
import math
from dataclasses import dataclass

from chatchtg.context.baseline import SYSTEM_PROMPT, BaselineComposer
from chatchtg.context.composer import ComposerInput
from chatchtg.context.types import Context, ImagePart, TextPart
from chatchtg.extractions.models import Extraction, Item
from chatchtg.objects.counter import PRODUCER, PRODUCER_VERSION, THRESHOLDS
from chatchtg.objects.questions import ObjectQuestion, classify

RASTER_TYPES = frozenset({"image/png", "image/jpeg"})

DOUBT_FRACTION = 0.5
"""A detection that scores at least this fraction of its class's minimum, but less than the
minimum, is shown as a doubtful candidate. Chosen on the valid split: at half the minimum,
the true count lies between the counted and the counted plus the doubtful candidates on 92%
of the window questions (61% are exact without them), with about 2 doubtful windows per plan."""

MIN_LONG_EDGE = 1400
"""A smaller drawing is enlarged to this long edge before the marks are drawn, so that the
numbers are readable without covering the symbols."""
MAX_LONG_EDGE = 2576
MAX_PIXELS = 3_750_000
"""The largest image that the API takes without downscaling (Opus 4.7 and later). A larger
drawing is reduced to it before the marks are drawn, so that the marks keep their width."""

COUNTED_COLOUR = (220, 0, 140)
DOUBTFUL_COLOUR = (0, 110, 230)
OUTLINE = 2
"""Line width of a mark, in pixels of the marked image."""
GAP = 3
"""Distance between a mark and the detected box, so that the outline does not cover the symbol."""
FONT_SIZE = 15

_NOUNS = {
    "window": "windows",
    "door": "doors",
    "toilet": "toilet bowls",
    "sink": "sinks and wash basins",
    "bath_shower": "showers and bathtubs",
}

HINT_SYSTEM_PROMPT = (
    "You answer questions about one construction drawing. Use only what the drawing shows. "
    "An object detector has marked candidates for the objects that the question counts. "
    "Per object it is right about 9 times in 10, so on a large plan a few marks are usually "
    "wrong: it misses objects, it marks things that are not the object, and it can mark one "
    "object twice (for example a window drawn as several panes, which is one window). "
    "Doubtful candidates scored below the detector's own threshold; many of them are real "
    "objects, many are not. "
    "Check every candidate on the drawing, and look over the whole drawing for objects that "
    "have no mark. Count by the rules in the question. "
    "Keep the answer short: name only the candidates you reject and the objects you add, a "
    "few words each, with the reason. End with the count, as 'Answer: <number>'."
)


@dataclass(frozen=True)
class Candidate:
    number: int
    item: Item
    doubtful: bool


def candidates(extraction: Extraction, question: ObjectQuestion) -> list[Candidate]:
    """The detections of the asked classes that are counted or doubtful, numbered in reading
    order (top to bottom, then left to right)."""
    chosen = []
    for item in extraction.items:
        if item.kind != "object" or item.label not in question.classes:
            continue
        score = item.score if item.score is not None else 1.0
        minimum = THRESHOLDS[item.label]
        if score >= minimum * DOUBT_FRACTION:
            chosen.append((item, score < minimum))
    chosen.sort(key=lambda pair: (round((pair[0].box[1] + pair[0].box[3]) / 2 / 20), pair[0].box[0]))
    return [Candidate(number, item, doubtful) for number, (item, doubtful) in enumerate(chosen, 1)]


def marked_scale(width: int, height: int) -> float:
    """The factor from drawing pixels to pixels of the marked image."""
    scale = max(1.0, MIN_LONG_EDGE / max(width, height))
    return min(scale, MAX_LONG_EDGE / max(width, height), math.sqrt(MAX_PIXELS / (width * height)))


def _label(candidate: Candidate) -> str:
    return f"{candidate.number}?" if candidate.doubtful else str(candidate.number)


def _tag_place(
    box: tuple[float, float, float, float],
    width: float,
    height: float,
    image_size: tuple[int, int],
    placed: list[tuple[float, float, float, float]],
) -> tuple[float, float]:
    """Where the number of a mark goes: outside the box, at one of its four corners, at the
    first corner where it overlaps no number placed before (above left when all overlap)."""
    x0, y0, x1, y1 = box
    corners = [(x0, y0 - height), (x1 - width, y0 - height), (x0, y1), (x1 - width, y1)]
    # Beside the box when it is narrow, for marks that overlap each other.
    corners += [(x1, y0), (x0 - width, y0), (x1, y1 - height), (x0 - width, y1 - height)]

    def inside(x: float, y: float) -> tuple[float, float]:
        return min(max(0.0, x), image_size[0] - width), min(max(0.0, y), image_size[1] - height)

    for x, y in corners:
        x, y = inside(x, y)
        if not any(x < b[2] and b[0] < x + width and y < b[3] and b[1] < y + height for b in placed):
            return x, y
    return inside(*corners[0])


def draw_marks(content: bytes, marks: list[Candidate]) -> tuple[bytes, float]:
    """The drawing as PNG with each candidate outlined and numbered, and its scale factor."""
    from PIL import Image, ImageDraw, ImageFont

    with Image.open(io.BytesIO(content)) as source:
        image = source.convert("RGB")
    scale = marked_scale(*image.size)
    if scale != 1.0:
        size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        image = image.resize(size, Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=FONT_SIZE)
    placed: list[tuple[float, float, float, float]] = []
    for candidate in marks:
        colour = DOUBTFUL_COLOUR if candidate.doubtful else COUNTED_COLOUR
        x0, y0, x1, y1 = (v * scale for v in candidate.item.box)
        box = (x0 - GAP - OUTLINE, y0 - GAP - OUTLINE, x1 + GAP + OUTLINE, y1 + GAP + OUTLINE)
        draw.rectangle(box, outline=colour, width=OUTLINE)
        text = _label(candidate)
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        tag_w, tag_h = right - left + 4, bottom - top + 4
        tag_x, tag_y = _tag_place(box, tag_w, tag_h, image.size, placed)
        placed.append((tag_x, tag_y, tag_x + tag_w, tag_y + tag_h))
        draw.rectangle((tag_x, tag_y, tag_x + tag_w, tag_y + tag_h), fill=colour)
        draw.text((tag_x + 2 - left, tag_y + 2 - top), text, fill=(255, 255, 255), font=font)
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue(), scale


def hint_text(question: ObjectQuestion, marks: list[Candidate], scale: float) -> str:
    noun = _NOUNS[question.kind]
    counted = [c for c in marks if not c.doubtful]
    if not marks:
        return (
            f"The detector found no candidates for {noun} on this drawing. Look over the "
            "drawing yourself."
        )
    lines = [
        "The first image is the drawing. The second image is the same drawing"
        + (f", scaled by {scale:.2f}," if scale != 1.0 else "")
        + f" with the detector's candidates for {noun} outlined and numbered: "
        "magenta numbers are candidates the detector counts, blue numbers with '?' are "
        "doubtful candidates. Positions are the centres in pixels of the second image.",
        f"Candidates ({len(counted)} counted, {len(marks) - len(counted)} doubtful):",
    ]
    for candidate in marks:
        item = candidate.item
        x0, y0, x1, y1 = (v * scale for v in item.box)
        score = f"{item.score:.2f}" if item.score is not None else "-"
        status = "doubtful" if candidate.doubtful else "counted"
        lines.append(
            f"{_label(candidate)}: {item.label}, score {score}, {status}, at ({round((x0 + x1) / 2)}, "
            f"{round((y0 + y1) / 2)})"
        )
    lines.append(f"The detector alone would answer {len(counted)}.")
    return "\n".join(lines)


logger = logging.getLogger(__name__)


class ObjectHintsComposer:
    name = "object-hints"
    version = 1
    reads = ((PRODUCER, PRODUCER_VERSION),)

    def __init__(self) -> None:
        self._baseline = BaselineComposer()

    def compose(self, request: ComposerInput) -> Context:
        question = classify(request.message)
        extraction = next(
            (
                e
                for e in request.extractions
                if (e.producer, e.version) == (PRODUCER, PRODUCER_VERSION)
            ),
            None,
        )
        if (
            question is None
            or extraction is None
            or request.document.media_type not in RASTER_TYPES
        ):
            return self._baseline.compose(request)
        marks = candidates(extraction, question)
        plain = ImagePart(media_type=request.document.media_type, data=request.content)
        if not marks:
            return Context(
                system=SYSTEM_PROMPT,
                parts=[plain, TextPart(text=hint_text(question, marks, 1.0)), TextPart(text=request.message)],
            )
        try:
            marked, scale = draw_marks(request.content, marks)
        except Exception:  # an image that cannot be drawn on is answered as in the baseline
            logger.warning("The marks could not be drawn on %s", request.document.filename, exc_info=True)
            return self._baseline.compose(request)
        return Context(
            system=HINT_SYSTEM_PROMPT,
            parts=[
                plain,
                ImagePart(media_type="image/png", data=marked),
                TextPart(text=hint_text(question, marks, scale)),
                TextPart(text=request.message),
            ],
        )
