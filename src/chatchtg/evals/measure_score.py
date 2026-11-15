"""Why a measured room is wrong: its outline compared with the DXF outline.

For scoring and error analysis only (ADR 0003): nothing here may be called at answer
time. Needs the dev dependency ``shapely``.
"""

from __future__ import annotations

import statistics

TOLERANCE = 0.02
"""A measured area counts as right within this share of the reference area."""
SLACK_PX = 1.5
"""An outline this close to the reference is an edge that is misplaced, not a piece that
is lost or added."""
MIN_EDGE_PX = 8.0

Polygon = list[tuple[float, float]]


def _shape(points: Polygon):
    from shapely.geometry import Polygon as Shape

    shape = Shape(points)
    return shape if shape.is_valid else shape.buffer(0)


def failure_pattern(
    truth_polygon: Polygon, truth_area: float, measured_polygon: Polygon, measured_area: float | None
) -> str:
    """ "ok", "no-area", "lost", "leak", "both" or "edges".

    "lost": a part of the reference room is missing from the measured region. "leak": the
    measured region holds space outside the reference room. "both": both. "edges": the
    region is the room, but its outline is misplaced by a pixel or so.
    """
    if measured_area is None or len(measured_polygon) < 3:
        return "no-area"
    if abs(measured_area - truth_area) <= TOLERANCE * truth_area:
        return "ok"
    truth, measured = _shape(truth_polygon), _shape(measured_polygon)
    if truth.area <= 0:
        return "edges"
    lost = truth.difference(measured.buffer(SLACK_PX)).area / truth.area > TOLERANCE
    leak = measured.difference(truth.buffer(SLACK_PX)).area / truth.area > TOLERANCE
    if lost and leak:
        return "both"
    if lost:
        return "lost"
    if leak:
        return "leak"
    return "edges"


def edge_offsets(truth_polygon: Polygon, measured_polygon: Polygon) -> list[tuple[float, float]]:
    """(length in px, median signed distance in px) for each reference edge of 8 px or more.

    The distance is from the reference edge to the measured outline. It is negative when
    the measured outline lies inside the reference edge.
    """
    from shapely.geometry import Point

    if len(measured_polygon) < 3:
        return []
    measured = _shape(measured_polygon)
    outline = measured.boundary
    offsets = []
    count = len(truth_polygon)
    for index in range(count):
        (x0, y0), (x1, y1) = truth_polygon[index], truth_polygon[(index + 1) % count]
        length = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        if length < MIN_EDGE_PX:
            continue
        distances = []
        for step in range(7):
            t = 0.2 + 0.1 * step
            point = Point(x0 + t * (x1 - x0), y0 + t * (y1 - y0))
            distance = outline.distance(point)
            distances.append(distance if measured.contains(point) else -distance)
        offsets.append((float(length), float(statistics.median(distances))))
    return offsets
