"""Read the windows, doors and fixtures of a CubiCasa5K ``model.svg`` as boxes.

The coordinates of the file are pixels of ``F1_scaled.png`` of the same plan.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass

from object_detector.classes import FIXTURES

Matrix = tuple[float, float, float, float, float, float]
"""An SVG matrix(a, b, c, d, e, f): x' = a*x + c*y + e, y' = b*x + d*y + f."""
Point = tuple[float, float]

IDENTITY: Matrix = (1, 0, 0, 1, 0, 0)
NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
PATH_TOKEN = re.compile(r"[MmLlHhVvQqZz]|" + NUMBER.pattern)
# The number of values that a path command takes.
PATH_ARITY = {"M": 2, "L": 2, "H": 1, "V": 1, "Q": 4, "Z": 0}


@dataclass(frozen=True)
class Box:
    label: str
    left: float
    top: float
    right: float
    bottom: float


def read_boxes(svg_text: str) -> list[Box]:
    """Return one box for each window, door and known fixture of all floors of the file."""
    boxes: list[Box] = []
    _walk(ET.fromstring(svg_text), IDENTITY, boxes)
    return boxes


def _walk(element: ET.Element, matrix: Matrix, boxes: list[Box]) -> None:
    # A file can hold several floors, and all but one have "display: none". They are read
    # too: the image shows all floors side by side, and the benchmark counts all of them.
    matrix = _compose(matrix, _matrix(element.get("transform")))
    identifier = element.get("id")
    words = element.get("class", "").split()
    if identifier == "Window":
        _add(boxes, "window", _own_polygon(element, matrix))
        return
    if identifier == "Door":
        # "Door None" is an opening that has no door leaf. It is not a door.
        if "None" not in words[1:2]:
            _add(boxes, "door", _own_polygon(element, matrix) + _door_swing(element, matrix))
        return
    if words[:1] == ["FixedFurniture"]:
        label = FIXTURES.get(words[1]) if len(words) > 1 else None
        if label:
            _add(boxes, label, _fixture_points(element, matrix))
        return
    for child in element:
        _walk(child, matrix, boxes)


def _add(boxes: list[Box], label: str, points: list[Point]) -> None:
    if points:
        xs = [x for x, _ in points]
        ys = [y for _, y in points]
        boxes.append(Box(label, min(xs), min(ys), max(xs), max(ys)))


def _matrix(transform: str | None) -> Matrix:
    if not transform or not transform.strip().startswith("matrix("):
        return IDENTITY
    values = [float(value) for value in NUMBER.findall(transform)]
    return tuple(values[:6]) if len(values) >= 6 else IDENTITY  # type: ignore[return-value]


def _compose(outer: Matrix, inner: Matrix) -> Matrix:
    """The matrix that applies ``inner`` first and ``outer`` second."""
    a1, b1, c1, d1, e1, f1 = outer
    a2, b2, c2, d2, e2, f2 = inner
    return (
        a1 * a2 + c1 * b2,
        b1 * a2 + d1 * b2,
        a1 * c2 + c1 * d2,
        b1 * c2 + d1 * d2,
        a1 * e2 + c1 * f2 + e1,
        b1 * e2 + d1 * f2 + f1,
    )


def _apply(matrix: Matrix, points: list[Point]) -> list[Point]:
    a, b, c, d, e, f = matrix
    return [(a * x + c * y + e, b * x + d * y + f) for x, y in points]


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _polygon_points(polygon: ET.Element) -> list[Point]:
    values = [float(value) for value in NUMBER.findall(polygon.get("points", ""))]
    return list(zip(values[0::2], values[1::2]))


def _own_polygon(element: ET.Element, matrix: Matrix) -> list[Point]:
    """The first polygon child: the opening of a window or a door in the wall."""
    for child in element:
        if _local(child.tag) == "polygon":
            return _apply(matrix, _polygon_points(child))
    return []


def _door_swing(door: ET.Element, matrix: Matrix) -> list[Point]:
    """The points of the drawn door leaves and swing arcs, so that the box covers the symbol."""
    points: list[Point] = []
    for panel in door:
        if panel.get("id") != "Panel":
            continue
        panel_matrix = _compose(matrix, _matrix(panel.get("transform")))
        for child in panel:
            if _local(child.tag) == "path":
                points += _apply(panel_matrix, _path_points(child.get("d", "")))
    return points


def _path_points(d: str) -> list[Point]:
    """The end points of the segments of a path of lines and quadratic curves.

    The control points are left out: the arc of a door stays inside the box of its end points.
    A path with another command gives no points.
    """
    if re.search(r"[^MmLlHhVvQqZz0-9eE\s,.+-]", d):
        return []
    tokens = PATH_TOKEN.findall(d)
    points: list[Point] = []
    x = y = 0.0
    start = (0.0, 0.0)
    index = 0
    command = ""
    while index < len(tokens):
        if tokens[index].isalpha():
            command = tokens[index]
            index += 1
        elif not command:
            return []
        arity = PATH_ARITY[command.upper()]
        if arity == 0:
            x, y = start
            command = ""  # a number after Z has no command: the path is not read
            continue
        if index + arity > len(tokens) or any(t.isalpha() for t in tokens[index : index + arity]):
            return []
        values = [float(token) for token in tokens[index : index + arity]]
        index += arity
        relative = command.islower()
        kind = command.upper()
        if kind == "H":
            x = x + values[0] if relative else values[0]
        elif kind == "V":
            y = y + values[0] if relative else values[0]
        else:
            # M and L have one point. Q has a control point and then the end point.
            end_x, end_y = values[-2], values[-1]
            x, y = (x + end_x, y + end_y) if relative else (end_x, end_y)
        if kind == "M":
            start = (x, y)
            command = "l" if relative else "L"  # more pairs after M are lines
        points.append((x, y))
    return points


def _fixture_points(fixture: ET.Element, matrix: Matrix) -> list[Point]:
    """The corners of the fixture's boundary polygon, or of all its shapes if it has none."""
    for child in fixture:
        if "BoundaryPolygon" in child.get("class", "").split():
            child_matrix = _compose(matrix, _matrix(child.get("transform")))
            points = [
                point
                for polygon in child.iter()
                if _local(polygon.tag) == "polygon"
                for point in _apply(child_matrix, _polygon_points(polygon))
            ]
            if points:
                return points
    return _shape_points(fixture, matrix, top=True)


def _shape_points(element: ET.Element, matrix: Matrix, top: bool = False) -> list[Point]:
    if not top:
        matrix = _compose(matrix, _matrix(element.get("transform")))
    tag = _local(element.tag)
    if tag == "polygon":
        return _apply(matrix, _polygon_points(element))
    if tag == "rect":
        try:
            x, y = float(element.get("x", 0)), float(element.get("y", 0))
            width, height = float(element.get("width", 0)), float(element.get("height", 0))
        except ValueError:
            return []
        return _apply(matrix, [(x, y), (x + width, y), (x + width, y + height), (x, y + height)])
    if tag == "path":
        return _apply(matrix, _path_points(element.get("d", "")))
    return [point for child in element for point in _shape_points(child, matrix)]
