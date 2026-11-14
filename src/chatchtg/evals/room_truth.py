"""Ground truth for room areas, read from the DXF twin of a Swiss Dwellings floor plan.

The DXF is used only to score and debug the area pipeline (ADR 0003): nothing here may be
called at answer time. Needs the dev dependencies ``ezdxf`` and ``shapely``.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

DPI = 150.0
NUMBER = re.compile(r"^[A-Z]?\d+\.\d+$")
AREA_TEXT = re.compile(r"^([\d.,]+)\s*m")
UNIT_TAG = re.compile(r"^(?:Whg\.|App\.)\s*([A-Z]+)$")


@dataclass
class TruthRoom:
    number: str
    name: str | None
    area: float
    """Reference area in m²: the outline minus the unnumbered polygons inside it."""
    printed_area: float | None
    apartment: str | None
    label_px: tuple[float, float]
    """Position of the room number text in the PNG."""
    polygon_px: list[tuple[float, float]]


@dataclass
class TruthPlan:
    stem: str
    scale: int
    pixels_per_metre: float
    offset_px: tuple[float, float]
    fit: float
    rooms: list[TruthRoom] = field(default_factory=list)
    tags_px: dict[str, tuple[float, float]] = field(default_factory=dict)
    text_boxes: list[tuple[float, float, float, float]] = field(default_factory=list)
    """Approximate pixel box (x0, y0, x1, y1) of every label text in the plan."""

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict) -> TruthPlan:
        rooms = [
            TruthRoom(
                **{
                    **room,
                    "label_px": tuple(room["label_px"]),
                    "polygon_px": [tuple(p) for p in room["polygon_px"]],
                }
            )
            for room in data["rooms"]
        ]
        tags = {k: tuple(v) for k, v in data["tags_px"].items()}
        boxes = [tuple(box) for box in data.get("text_boxes", [])]
        return cls(
            **{**data, "rooms": rooms, "tags_px": tags, "offset_px": tuple(data["offset_px"]),
               "text_boxes": boxes}
        )


def _to_px(points: np.ndarray, transform: tuple[float, float, float]) -> np.ndarray:
    s, tx, ty = transform
    return np.stack([points[:, 0] * s + tx, -points[:, 1] * s + ty], axis=1)


def _raster(polygons: list[np.ndarray], transform, shape, shift: int = 6) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    f = 1 << shift
    cv2.fillPoly(
        mask, [np.round(_to_px(p, transform) * f).astype(np.int32) for p in polygons], 1, shift=shift
    )
    return mask


def fit_transform(image: np.ndarray, walls: list[np.ndarray], scale: int):
    """Fit px = s*x + tx, py = -s*y + ty by matching the DXF wall polygons to dark ink.

    The scale is the nominal one of the title block; only the offset is fitted.
    """
    height, width = image.shape[:2]
    grey = image.mean(axis=2)
    red = (image[..., 0].astype(int) - image[..., 2].astype(int)) > 60
    ink = ((grey < 200) & ~red).astype(np.float32)
    s = 1000.0 / scale / 25.4 * DPI
    points = np.concatenate(walls)
    cx, cy = points[:, 0].mean(), points[:, 1].mean()
    start = (s, width / 2 - s * cx, height / 2 + s * cy)
    wall_mask = _raster(walls, start, (height, width), shift=4).astype(np.float32)
    spectrum = np.fft.rfft2(ink) * np.conj(np.fft.rfft2(wall_mask))
    corr = np.fft.irfft2(spectrum, s=ink.shape)
    dy, dx = np.unravel_index(np.argmax(corr), corr.shape)
    if dy > height / 2:
        dy -= height
    if dx > width / 2:
        dx -= width
    best_t = (s, start[1] + dx, start[2] + dy)

    def score(t):
        mask = _raster(walls, t, (height, width))
        return float((ink * mask).sum() / (mask.sum() + 1e-9))

    best = score(best_t)
    for step in (1.0, 0.5, 0.25, 0.125):
        improved = True
        while improved:
            improved = False
            for ddx, ddy in ((step, 0), (-step, 0), (0, step), (0, -step)):
                t = (s, best_t[1] + ddx, best_t[2] + ddy)
                value = score(t)
                if value > best + 1e-6:
                    best, best_t, improved = value, t, True
    return best_t, best


def read_truth(dxf_path: Path, image: np.ndarray) -> TruthPlan:
    """Read the rooms of one plan. ``image`` is the PNG twin as an RGB array."""
    import ezdxf
    from shapely.geometry import Point, Polygon

    model = ezdxf.readfile(dxf_path).modelspace()

    def polylines(layer: str) -> list[np.ndarray]:
        found = []
        for entity in model.query(f'LWPOLYLINE[layer=="{layer}"]'):
            points = np.array([(p[0], p[1]) for p in entity.get_points()], float)
            if len(points) >= 3:
                found.append(points)
        return found

    scale = 100
    texts = []
    heights: list[float] = []
    for entity in model.query("TEXT"):
        text = entity.dxf.text
        if entity.dxf.layer == "ANNOTATION":
            match = re.search(r"1:(\d+)", text)
            if match:
                scale = int(match.group(1))
        elif entity.dxf.layer == "ROOM_TEXT":
            texts.append((text, float(entity.dxf.insert[0]), float(entity.dxf.insert[1])))
            heights.append(float(entity.dxf.height))

    transform, fit = fit_transform(image, polylines("WALL"), scale)

    def valid(points):
        polygon = Polygon(points)
        return polygon if polygon.is_valid else polygon.buffer(0)

    rooms = [
        {"points": p, "poly": valid(p), "area": Polygon(p).area, "name": None, "number": None,
         "printed": None, "label": None}
        for p in polylines("ROOM_OUTLINE")
    ]
    units = [valid(p) for p in polylines("UNIT_OUTLINE")]
    tags: list[tuple[str, Point]] = []
    for text, x, y in texts:
        point = Point(x, y)
        tag = UNIT_TAG.match(text)
        if tag:
            tags.append((tag.group(1), point))
            continue
        hosts = [room for room in rooms if room["poly"].contains(point)]
        if not hosts:
            continue
        room = min(hosts, key=lambda r: r["area"])
        if NUMBER.match(text):
            room["number"], room["label"] = text, (x, y)
        elif AREA_TEXT.match(text):
            room["printed"] = float(AREA_TEXT.match(text).group(1).replace(",", "."))
        else:
            room["name"] = text
    # An unnumbered polygon inside a numbered room (a column, a shaft) is a hole of that room.
    holes = []
    for hole in sorted(rooms, key=lambda r: -r["area"]):
        if hole["number"] or hole["name"]:
            continue
        if any(o["poly"].intersection(hole["poly"]).area > 0.9 * hole["poly"].area for o in holes):
            continue
        for room in rooms:
            if room is hole or not room["number"]:
                continue
            if room["poly"].intersection(hole["poly"]).area > 0.9 * hole["poly"].area:
                room["area"] -= hole["area"]
                holes.append(hole)
                break
    unit_letter: dict[int, str] = {}
    for letter, point in tags:
        for index, unit in enumerate(units):
            if unit.contains(point):
                unit_letter[index] = letter

    plan = TruthPlan(
        stem=dxf_path.stem, scale=scale, pixels_per_metre=transform[0],
        offset_px=(transform[1], transform[2]), fit=fit,
    )
    for (text, x, y), text_height in zip(texts, heights):
        cx, cy = _to_px(np.array([[x, y]]), transform)[0]
        h = text_height * transform[0]
        w = (0.62 * len(text) + 0.6) * h
        plan.text_boxes.append((float(cx - w / 2), float(cy - 0.75 * h), float(cx + w / 2), float(cy + 0.75 * h)))
    for letter, point in tags:
        x, y = _to_px(np.array([[point.x, point.y]]), transform)[0]
        plan.tags_px[letter] = (float(x), float(y))
    for room in rooms:
        if not room["number"]:
            continue
        apartment, best = None, 0.5
        for index, unit in enumerate(units):
            share = room["poly"].intersection(unit).area / max(room["poly"].area, 1e-9)
            if share > best:
                apartment, best = unit_letter.get(index), share
        label = _to_px(np.array([room["label"]]), transform)[0]
        plan.rooms.append(
            TruthRoom(
                number=room["number"], name=room["name"], area=float(room["area"]),
                printed_area=room["printed"], apartment=apartment,
                label_px=(float(label[0]), float(label[1])),
                polygon_px=[(float(x), float(y)) for x, y in _to_px(room["points"], transform)],
            )
        )
    return plan
