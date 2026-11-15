"""Read the doors, windows and fixtures of a Swiss Dwellings DXF twin as boxes in PNG pixels.

The DXF files are training labels and a reference for scores only. The detector
that is trained on these boxes sees only the PNG.

The counting rules reproduce the answers of the benchmark on all 1,040 plans (see
``docs/benchmarks/2026-11-15-swiss-symbol-detector.md``). They are fitted to the one
renderer that drew these plans:

- door: one open 5-point polyline on the layer DOOR (the opening in the wall). A double door
  has one opening and two swing arcs, and is one door. The box covers the opening and its arcs.
- window: one closed 4-point polyline on the layer WINDOW. The two 2-point lines in it are
  the glass, not more windows.
- On the layer FIXTURE, each object is a closed 4-point outline with marks inside:
  - toilet: a 48-point bowl without a circle inside it;
  - washbasin: a 48-point bowl with a drain circle inside it;
  - bathtub: a drain circle of radius 0.025 m that is not in a 48-point bowl (most bathtubs
    also have a 69-point inner line, a few narrow ones have none);
  - shower: a 5-point polyline (the tray mark);
  - washing machine: an outline whose only mark is a circle of radius 0.15 to 0.3 m. It is
    not asked about; it is a class so that the detector does not take it for a washbasin.
  The box of a fixture is the smallest outline that holds its mark.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from object_detector.cubicasa_svg import Box

# The order is the category id in the dataset and in the model.
SWISS_CLASSES = ["door", "window", "toilet", "washbasin", "bathtub", "shower", "washing machine"]
# The classes that the benchmark asks about.
ASKED_CLASSES = SWISS_CLASSES[:6]

DPI = 150.0
"""The PNG files are rendered at 150 dots per inch."""
TOLERANCE = 0.02
"""Metres: a mark is inside an outline if it is inside the outline grown by this much."""
BATHTUB_DRAIN = 0.025
HOB_RING = 0.08
"""The radius of a cooking ring. It is not a drain."""
WASHING_MACHINE_DRUM = (0.15, 0.3)


@dataclass(frozen=True)
class Polyline:
    layer: str
    points: tuple[tuple[float, float], ...]
    closed: bool


@dataclass(frozen=True)
class Circle:
    layer: str
    center: tuple[float, float]
    radius: float


@dataclass(frozen=True)
class Arc:
    layer: str
    center: tuple[float, float]
    radius: float
    start: float
    """Degrees, counter-clockwise from the x axis."""
    end: float


@dataclass
class Drawing:
    """The entities of one DXF that the labels need, in metres (y up)."""

    scale: int = 100
    polylines: list[Polyline] = field(default_factory=list)
    circles: list[Circle] = field(default_factory=list)
    arcs: list[Arc] = field(default_factory=list)


@dataclass(frozen=True)
class Transform:
    """px = scale * x + dx, py = -scale * y + dy."""

    scale: float
    dx: float
    dy: float

    def apply(self, box: Box) -> Box:
        left, right = box.left * self.scale + self.dx, box.right * self.scale + self.dx
        top, bottom = -box.bottom * self.scale + self.dy, -box.top * self.scale + self.dy
        return Box(box.label, left, top, right, bottom)


def read_drawing(path: Path) -> Drawing:
    """Read a DXF file with ezdxf (a dev dependency)."""
    import ezdxf

    drawing = Drawing()
    for entity in ezdxf.readfile(path).modelspace():
        layer, kind = entity.dxf.layer, entity.dxftype()
        if kind == "LWPOLYLINE":
            points = tuple((float(p[0]), float(p[1])) for p in entity.get_points())
            drawing.polylines.append(Polyline(layer, points, bool(entity.closed)))
        elif kind == "CIRCLE":
            center = entity.dxf.center
            drawing.circles.append(Circle(layer, (center[0], center[1]), float(entity.dxf.radius)))
        elif kind == "ARC":
            center = entity.dxf.center
            drawing.arcs.append(
                Arc(layer, (center[0], center[1]), float(entity.dxf.radius),
                    float(entity.dxf.start_angle), float(entity.dxf.end_angle))
            )
        elif kind == "TEXT" and layer == "ANNOTATION":
            match = re.search(r"1:(\d+)", entity.dxf.text)
            if match:
                drawing.scale = int(match.group(1))
    return drawing


def _bounds(points) -> tuple[float, float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def _inside(inner: tuple[float, float, float, float], outer: tuple[float, float, float, float]) -> bool:
    return (
        inner[0] >= outer[0] - TOLERANCE and inner[1] >= outer[1] - TOLERANCE
        and inner[2] <= outer[2] + TOLERANCE and inner[3] <= outer[3] + TOLERANCE
    )


def _arc_bounds(arc: Arc) -> tuple[float, float, float, float]:
    """The bounds of the swing: the hinge and the drawn arc."""
    end = arc.end if arc.end > arc.start else arc.end + 360
    steps = max(2, math.ceil((end - arc.start) / 5))
    points = [arc.center] + [
        (arc.center[0] + arc.radius * math.cos(math.radians(a)),
         arc.center[1] + arc.radius * math.sin(math.radians(a)))
        for a in np.linspace(arc.start, end, steps + 1)
    ]
    return _bounds(points)


def _union(a, b):
    return min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])


def read_symbols(drawing: Drawing) -> list[Box]:
    """One box in metres (top < bottom in y up, so ``top`` is the lower y) for each object."""
    boxes: list[Box] = []

    def add(label: str, bounds) -> None:
        boxes.append(Box(label, *bounds))

    # Doors: one opening each, grown by the swing arcs whose hinge is on it.
    opening_lines = [
        p.points for p in drawing.polylines
        if p.layer == "DOOR" and len(p.points) == 5 and not p.closed
    ]
    door_bounds = [_bounds(points) for points in opening_lines]
    for arc in drawing.arcs:
        if arc.layer != "DOOR" or not opening_lines:
            continue
        # The hinge is a corner of its opening. The bounds of a slanted opening can hold
        # the hinge of a neighbour, so the nearest corner decides.
        distances = [
            min(math.dist(arc.center, corner) for corner in points) for points in opening_lines
        ]
        index = int(np.argmin(distances))
        if distances[index] <= 0.05:
            door_bounds[index] = _union(door_bounds[index], _arc_bounds(arc))
    for bounds in door_bounds:
        add("door", bounds)

    for p in drawing.polylines:
        if p.layer == "WINDOW" and p.closed and len(p.points) == 4:
            add("window", _bounds(p.points))

    outlines = [
        _bounds(p.points) for p in drawing.polylines
        if p.layer == "FIXTURE" and p.closed and len(p.points) == 4
    ]

    def outline_of(mark) -> tuple[float, float, float, float]:
        holders = [o for o in outlines if _inside(mark, o)]
        if not holders:
            return mark
        return min(holders, key=lambda o: (o[2] - o[0]) * (o[3] - o[1]))

    circles = [c for c in drawing.circles if c.layer == "FIXTURE"]

    def circle_bounds(c: Circle):
        return c.center[0] - c.radius, c.center[1] - c.radius, c.center[0] + c.radius, c.center[1] + c.radius

    bowls = [
        _bounds(p.points) for p in drawing.polylines
        if p.layer == "FIXTURE" and len(p.points) == 48 and not p.closed
    ]
    drains = [c for c in circles if c.radius < 0.2 and abs(c.radius - HOB_RING) > 0.001]
    for bowl in bowls:
        has_drain = any(_inside((*c.center, *c.center), bowl) for c in drains)
        add("washbasin" if has_drain else "toilet", outline_of(bowl))
    for c in circles:
        point = (*c.center, *c.center)
        if abs(c.radius - BATHTUB_DRAIN) < 0.0006 and not any(_inside(point, b) for b in bowls):
            add("bathtub", outline_of(circle_bounds(c)))
    for p in drawing.polylines:
        if p.layer == "FIXTURE" and len(p.points) == 5 and not p.closed:
            add("shower", outline_of(_bounds(p.points)))

    # A washing machine: an outline that holds a drum circle and no other mark.
    marks = [_bounds(p.points) for p in drawing.polylines
             if p.layer == "FIXTURE" and not (p.closed and len(p.points) == 4)]
    for c in circles:
        if not WASHING_MACHINE_DRUM[0] <= c.radius <= WASHING_MACHINE_DRUM[1]:
            continue
        outline = outline_of(circle_bounds(c))
        if not any(_inside(m, outline) for m in marks):
            add("washing machine", outline)
    return boxes


# --- DXF metres to PNG pixels -------------------------------------------------------------


def pixels_per_metre(scale: int) -> float:
    return 1000.0 / scale / 25.4 * DPI


def _ink(image: np.ndarray) -> np.ndarray:
    """Dark pixels that are not red (the red lines are apartment borders, not in the DXF)."""
    grey = image.mean(axis=2)
    red = (image[..., 0].astype(int) - image[..., 2].astype(int)) > 60
    return ((grey < 200) & ~red).astype(np.float32)


def _raster(polygons: list[np.ndarray], scale: float, dx: float, dy: float, shape) -> np.ndarray:
    from PIL import Image, ImageDraw

    mask = Image.new("L", (shape[1], shape[0]), 0)
    draw = ImageDraw.Draw(mask)
    for polygon in polygons:
        xy = [(x * scale + dx, -y * scale + dy) for x, y in polygon]
        if len(xy) >= 3:
            draw.polygon(xy, fill=1)
    return np.asarray(mask, dtype=np.float32)


def fit_transform(image: np.ndarray, drawing: Drawing) -> tuple[Transform, float]:
    """Find the offset that puts the walls of the DXF on the dark ink of the PNG.

    The scale is the nominal one of the title block (150 dpi), and only the offset is fitted,
    by a cross-correlation of the wall polygons with the ink, to the nearest pixel. This is
    the method of ``chatchtg.evals.room_truth.fit_transform`` without the sub-pixel step.
    Returns the transform and the share of wall pixels that are on ink (near 1 is good).
    """
    height, width = image.shape[:2]
    walls = [np.array(p.points) for p in drawing.polylines if p.layer == "WALL" and len(p.points) >= 3]
    if not walls:
        raise ValueError("The drawing has no walls to fit.")
    scale = pixels_per_metre(drawing.scale)
    points = np.concatenate(walls)
    cx, cy = points[:, 0].mean(), points[:, 1].mean()
    dx0, dy0 = width / 2 - scale * cx, height / 2 + scale * cy
    ink = _ink(image)
    wall_mask = _raster(walls, scale, dx0, dy0, ink.shape)
    spectrum = np.fft.rfft2(ink) * np.conj(np.fft.rfft2(wall_mask))
    correlation = np.fft.irfft2(spectrum, s=ink.shape)
    shift_y, shift_x = np.unravel_index(np.argmax(correlation), correlation.shape)
    if shift_y > height / 2:
        shift_y -= height
    if shift_x > width / 2:
        shift_x -= width
    transform = Transform(scale, dx0 + shift_x, dy0 + shift_y)
    mask = _raster(walls, scale, transform.dx, transform.dy, ink.shape)
    fit = float((ink * mask).sum() / max(mask.sum(), 1.0))
    return transform, fit


def boxes_in_pixels(boxes: list[Box], transform: Transform, size: tuple[int, int]) -> list[Box]:
    """Map boxes in metres to PNG pixels and cut them to the image of ``size`` (width, height)."""
    width, height = size
    out = []
    for box in boxes:
        b = transform.apply(box)
        out.append(Box(
            b.label,
            min(max(b.left, 0.0), width), min(max(b.top, 0.0), height),
            min(max(b.right, 0.0), width), min(max(b.bottom, 0.0), height),
        ))
    return out
