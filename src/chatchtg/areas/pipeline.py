"""The area pipeline: one plan image in, a table of rooms with areas out.

Stages, each from pixels only:

1. Labels: local OCR reads the scale, the room numbers, names and printed areas.
2. Membership: grey floor marks outdoor rooms, red outlines assign rooms to apartments.
3. Areas: a printed area is used where there is one, otherwise the room is measured.

Everything is deterministic and makes no LLM call, so a composer may use it.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import asdict, dataclass, field
from functools import lru_cache

import numpy as np

from chatchtg.areas.labels import PlanLabels, read_labels
from chatchtg.areas.measure import measure_rooms
from chatchtg.areas.membership import assign_apartments, outdoor_rooms
from chatchtg.areas.questions import RoomRow

DPI = 150.0
"""The benchmark plans are A3 sheets rendered at 150 dpi."""
DEFAULT_SCALE = 100
PRINTED_TOLERANCE = 0.25
"""A printed area further than this from a clean measurement is taken as misread."""


@dataclass
class PlanRoom:
    number: str
    name: str | None
    area: float | None
    """The area the pipeline stands behind, in m²."""
    method: str | None
    """"printed", "measured" or None when there is no area."""
    printed_area: float | None
    measured_area: float | None
    apartment: str | None
    outdoor: bool
    seed: tuple[float, float]
    polygon: list[tuple[float, float]] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)


@dataclass
class PlanAnalysis:
    scale: int
    scale_read: bool
    """False when the title block could not be read and the scale is a guess."""
    pixels_per_metre: float
    rooms: list[PlanRoom]
    missing_numbers: list[str]
    """Gaps in the room numbering: rooms whose label was probably not read."""

    def rows(self) -> list[RoomRow]:
        return [
            RoomRow(number=r.number, name=r.name, area=r.area, apartment=r.apartment, outdoor=r.outdoor)
            for r in self.rooms
        ]

    def prints_areas(self) -> bool:
        """True when the plan prints the area under most room labels."""
        printed = sum(room.printed_area is not None for room in self.rooms)
        return printed * 2 > len(self.rooms)

    def may_be_incomplete(self, kind: str) -> bool:
        """True when a room whose label was not read could change the answer to this kind
        of question: any total may lack it, a single room that was found does not."""
        return kind != "room" and bool(self.missing_numbers)

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict) -> PlanAnalysis:
        rooms = [
            PlanRoom(**{**room, "seed": tuple(room["seed"]), "polygon": [tuple(p) for p in room["polygon"]]})
            for room in data["rooms"]
        ]
        return cls(**{**data, "rooms": rooms})


def pixels_per_metre(scale: int) -> float:
    return DPI / 0.0254 / scale


def analyse_plan(image: np.ndarray, labels: PlanLabels | None = None) -> PlanAnalysis:
    """Analyse one plan. ``image`` is an RGB uint8 array."""
    labels = labels or read_labels(image)
    scale = labels.scale or DEFAULT_SCALE
    ppm = pixels_per_metre(scale)
    seeds = [room.seed for room in labels.rooms]
    points = [room.number_xy for room in labels.rooms]
    measures = measure_rooms(image, seeds, ppm, labels.text_boxes)
    apartments = assign_apartments(image, points, labels.tags)
    outdoor = outdoor_rooms(image, points)

    rooms = []
    for label, measure, apartment, is_outdoor in zip(labels.rooms, measures, apartments, outdoor):
        measured = measure.area_m2
        clean = measured is not None and not measure.flags
        area, method = None, None
        if label.printed_area is not None:
            area, method = label.printed_area, "printed"
            if clean and abs(label.printed_area - measured) > PRINTED_TOLERANCE * measured:
                area, method = measured, "measured"
        elif measured is not None:
            area, method = measured, "measured"
        rooms.append(
            PlanRoom(
                number=label.number, name=label.name, area=area, method=method,
                printed_area=label.printed_area, measured_area=measured, apartment=apartment,
                outdoor=bool(is_outdoor), seed=label.seed, polygon=measure.polygon, flags=list(measure.flags),
            )
        )
    return PlanAnalysis(
        scale=scale, scale_read=labels.scale is not None, pixels_per_metre=ppm, rooms=rooms,
        missing_numbers=list(labels.missing_numbers),
    )


def decode_image(content: bytes) -> np.ndarray:
    from PIL import Image

    return np.array(Image.open(io.BytesIO(content)).convert("RGB"))


@lru_cache(maxsize=32)
def _analyse_cached(digest: str, content: bytes) -> PlanAnalysis:
    return analyse_plan(decode_image(content))


def analyse_bytes(content: bytes) -> PlanAnalysis:
    """Analyse an encoded image. The result is kept in memory by the hash of the bytes,
    so the questions that share a drawing pay for the analysis once."""
    return _analyse_cached(hashlib.sha256(content).hexdigest(), content)
