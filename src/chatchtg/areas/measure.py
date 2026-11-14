"""Measure room areas on a rendered floor plan, from pixels only.

How a room is found:

1. Every drawn pixel ("ink") splits the white space into components.
2. Each room label claims the component it sits in.
3. Two components that touch across a line are joined when that line is not a room
   boundary. A room boundary is a wall or a thin light-grey separator. Everything else
   (fixtures, door leaves and swings, stair treads, lift crosses) is drawn inside a room.
   A component joins a room only when all its joinable neighbours belong to that room, so
   the pocket of a door or window opening, which lies between two rooms, joins neither.
4. The outline of the joined region is turned into a polygon. Each polygon edge is moved
   to the centre of the drawn line under it, found to a fraction of a pixel from the
   anti-aliased grey values. The reference areas are measured to that centre.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy import ndimage

K3 = np.ones((3, 3), np.uint8)
K5 = np.ones((5, 5), np.uint8)

WALL_GREY = 176
SEPARATOR_INTEGRAL = (85.0, 159.0)
"""Darkness across a room separator line, in grey levels times pixels (measured: 152 on
1:100 plans, 101 on 1:200 plans).
The next lightest line, the frame of a door, has 165."""
SEPARATOR_PEAK = 108.0
"""A separator is light grey: no pixel on it is darker than this."""
RED_DARKNESS_PER_SATURATION = 1.25
"""The red outline is (176, 58, 46): each unit of saturation comes with 1.25 of darkness."""
OUTDOOR_SHADE_MAX = 236.0
"""An outdoor room is shaded grey 228; a stairwell 243; other floors are white."""
HALF_WIDTH_BLACK = 0.95
"""Pixels from the inner edge of a wall outline to its centre."""
OUTLINE_WIDTH = 2.04
"""Measured width of a wall outline between the two edge thresholds, in pixels."""
FILL_EDGE = 167.0
"""Darkness halfway between a black outline (255) and the wall fill behind it (79)."""
HALF_WIDTH_THIN = 0.8
"""The same for a thin line whose centre cannot be read directly."""


def _disc(radius: float) -> np.ndarray:
    """An anti-aliased disc. The darkness summed over it, divided by the diameter, is the
    darkness across a straight line through its centre, whatever the line's direction."""
    size = int(np.ceil(radius)) * 2 + 1
    fine = 8
    axis = (np.arange(size * fine) + 0.5) / fine - size / 2
    inside = (axis[None, :] ** 2 + axis[:, None] ** 2) <= radius**2
    return inside.reshape(size, fine, size, fine).mean(axis=(1, 3)).astype(np.float32)


_DISC_RADIUS = 4.0
_DISC = _disc(_DISC_RADIUS)


@dataclass
class RoomMeasure:
    seed: tuple[float, float]
    area_m2: float | None = None
    polygon: list[tuple[float, float]] = field(default_factory=list)
    """Outer outline in pixel coordinates, after the edges were moved to the line centres."""
    holes: list[list[tuple[float, float]]] = field(default_factory=list)
    pixel_area_m2: float | None = None
    """Area of the raw pixel region, before the edges were moved. For diagnosis."""
    flags: list[str] = field(default_factory=list)
    """Reasons to distrust the area: "no-region", "shared-region", "open-region"."""
    box: tuple[int, int, int, int] | None = None
    shade: float | None = None
    """Median grey value of the room's empty floor: 255 is white, an outdoor room is darker."""


@dataclass
class Raster:
    """Pixel classes of one plan."""

    grey: np.ndarray
    dark: np.ndarray
    """Darkness against the local background, 0 to 255. Red and removed text are 0."""
    ink: np.ndarray
    wall: np.ndarray
    line: np.ndarray
    """Ink that is not wall."""
    joinable: np.ndarray
    """Line pixels that do not look like a room boundary."""
    red: np.ndarray
    opening: np.ndarray
    """Pixels in a gap of a straight wall."""


def classify(
    image: np.ndarray,
    text_boxes: list[tuple[float, float, float, float]] = (),
    pixels_per_metre: float | None = None,
) -> Raster:
    """Split an RGB plan into wall, line and free pixels.

    ``text_boxes`` are the pixel boxes (x0, y0, x1, y1) of the label texts. A label
    printed over a line would cut a small room into pieces, so the text is erased.
    """
    rgb = image.astype(np.int16)
    grey = image.astype(np.float32).mean(axis=2)
    # Everything is drawn in grey except the red apartment outline, so the colour
    # saturation of a pixel says how much of it the red stroke covers.
    saturation = (rgb.max(axis=2) - rgb.min(axis=2)).astype(np.float32)
    red = saturation > 6

    # Shaded floors (outdoor rooms, stairwells) are flat light grey. Measure darkness
    # against the local background so that a shaded floor counts as empty.
    background = cv2.dilate(grey, np.ones((7, 7), np.uint8))
    background = np.maximum(background, 215.0)
    dark = np.clip(background - grey, 0, None) * (255.0 / background)
    dark = np.clip(dark - RED_DARKNESS_PER_SATURATION * saturation, 0, None)

    fill = (np.abs(grey - WALL_GREY) <= 8) & ~red
    fill = cv2.morphologyEx(fill.astype(np.uint8), cv2.MORPH_OPEN, K3)
    wall_region = cv2.morphologyEx(fill, cv2.MORPH_CLOSE, K5).astype(bool)

    ink = dark > 20
    # A line hidden under the red stroke: restore it where ink continues on both sides.
    ink8 = ink.astype(np.uint8)
    across = cv2.morphologyEx(ink8, cv2.MORPH_CLOSE, np.ones((1, 9), np.uint8)) | cv2.morphologyEx(
        ink8, cv2.MORPH_CLOSE, np.ones((9, 1), np.uint8)
    )
    hidden = (saturation > 60) & across.astype(bool) & ~ink
    ink |= hidden
    dark[hidden] = 255.0 - 160.0

    # Erase the label texts, but keep walls and long straight lines that pass under them.
    if len(text_boxes):
        ink8 = ink.astype(np.uint8)
        keep = cv2.morphologyEx(ink8, cv2.MORPH_OPEN, np.ones((1, 26), np.uint8)) | cv2.morphologyEx(
            ink8, cv2.MORPH_OPEN, np.ones((26, 1), np.uint8)
        )
        black = ((grey < 128) & (dark > 127)).astype(np.uint8)
        solid = cv2.distanceTransform(black, cv2.DIST_L2, 3) >= 1.8
        keep = keep.astype(bool) | wall_region | cv2.dilate(solid.astype(np.uint8), K5).astype(bool)
        text = np.zeros(ink.shape, bool)
        for x0, y0, x1, y1 in text_boxes:
            text[max(0, int(y0) - 1) : int(y1) + 2, max(0, int(x0) - 1) : int(x1) + 2] = True
        erased = text & ink & ~keep
        ink &= ~erased
        dark[erased] = 0.0

    # Text and other small marks are not boundaries: remove ink islands away from walls.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    small = (np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT]) <= 30) & (
        np.arange(count) > 0
    )
    touches_wall = np.zeros(count, bool)
    touches_wall[np.unique(labels[wall_region])] = True
    island = (small & ~touches_wall)[labels]
    ink &= ~island
    dark[island] = 0.0

    integral = cv2.filter2D(dark, -1, _DISC / (2 * _DISC_RADIUS), borderType=cv2.BORDER_CONSTANT)
    # A wall too thin to show its hatch is a black line about four pixels thick; a door
    # leaf is two.
    black = ((grey < 128) & (dark > 127)).astype(np.uint8)
    core = cv2.distanceTransform(black, cv2.DIST_L2, 3) >= 1.8
    thick = cv2.dilate(core.astype(np.uint8), K5).astype(bool) & black.astype(bool)
    near_fill = cv2.dilate(wall_region.astype(np.uint8), K5).astype(bool)
    wall = wall_region | (near_fill & ink & (grey < 150)) | thick
    wall |= cv2.morphologyEx(wall.astype(np.uint8), cv2.MORPH_CLOSE, K5).astype(bool) & ink
    line = ink & ~wall
    separator = (integral >= SEPARATOR_INTEGRAL[0]) & (integral <= SEPARATOR_INTEGRAL[1])
    # The disc also picks up a second line that runs alongside within 4 px (a balcony
    # railing). Across a horizontal or vertical line, three pixels are enough.
    across = np.minimum(
        cv2.filter2D(dark, -1, np.ones((1, 3), np.float32), borderType=cv2.BORDER_CONSTANT),
        cv2.filter2D(dark, -1, np.ones((3, 1), np.float32), borderType=cv2.BORDER_CONSTANT),
    )
    separator |= (across >= SEPARATOR_INTEGRAL[0]) & (across <= SEPARATOR_INTEGRAL[1])
    separator &= cv2.dilate(dark, K5) <= SEPARATOR_PEAK
    separator &= line
    # A fixture drawn against a separator hides it. A separator is straight, so it carries
    # on along its row or column for as long as there is ink.
    for axis, shape in ((1, (1, 8)), (0, (8, 1))):
        stretch = cv2.morphologyEx(separator.astype(np.uint8), cv2.MORPH_OPEN, np.ones(shape, np.uint8))
        separator |= _extend_along(stretch.astype(bool), line, axis)
    joinable = line & ~separator & ~cv2.dilate(wall.astype(np.uint8), K3).astype(bool)
    opening = np.zeros(ink.shape, bool)
    if pixels_per_metre:
        opening = _wall_gaps(wall, pixels_per_metre)
        joinable &= ~opening
    return Raster(
        grey=grey, dark=dark, ink=ink, wall=wall, line=line, joinable=joinable, red=red, opening=opening
    )


def _extend_along(marker: np.ndarray, mask: np.ndarray, axis: int) -> np.ndarray:
    """The runs of ``mask`` along ``axis`` that hold a marker pixel."""
    if axis == 0:
        return _extend_along(marker.T, mask.T, 1).T
    previous = np.zeros_like(mask)
    previous[:, 1:] = mask[:, :-1]
    run = np.cumsum((mask & ~previous).ravel()).reshape(mask.shape)
    marked = np.zeros(int(run.max()) + 1, bool)
    marked[run[marker & mask]] = True
    return mask & marked[run]


def _wall_gaps(wall: np.ndarray, scale: float) -> np.ndarray:
    """Door openings in thin horizontal or vertical walls.

    In a wall only a few pixels thick the two lines that close a door opening touch, so
    the opening leaves no pocket to find. A thin wall segment is a run of wall pixels at
    least 0.4 m long and at most 6 pixels thick. Two segments in line, at most 1.3 m
    apart, have an opening between them.
    """
    wall8 = wall.astype(np.uint8)
    run = max(6, int(round(0.4 * scale)))
    thick = 7
    reach = int(round(1.3 * scale))
    gaps = np.zeros(wall.shape, bool)
    for horizontal in (True, False):
        along = (1, run) if horizontal else (run, 1)
        across = (thick, 1) if horizontal else (1, thick)
        bridge = (1, reach) if horizontal else (reach, 1)
        segment = cv2.morphologyEx(wall8, cv2.MORPH_OPEN, np.ones(along, np.uint8)) & ~cv2.morphologyEx(
            wall8, cv2.MORPH_OPEN, np.ones(across, np.uint8)
        )
        closed = cv2.morphologyEx(segment, cv2.MORPH_CLOSE, np.ones(bridge, np.uint8))
        gaps |= closed.astype(bool) & ~wall
    return gaps


def _contacts(labels: np.ndarray, raster: Raster, reach: int = 2):
    """Pairs of free components that face each other across a line.

    Returns {(a, b): (pixels, joinable_pixels)} with a < b.
    """
    height, width = labels.shape
    ys, xs = np.nonzero(raster.line)
    joinable = raster.joinable[ys, xs]
    keys, flags = [], []
    for dx in range(0, reach + 1):
        for dy in range(-reach, reach + 1):
            if dx == 0 and dy <= 0:
                continue
            y1, x1, y2, x2 = ys + dy, xs + dx, ys - dy, xs - dx
            ok = (y1 >= 0) & (y1 < height) & (x1 < width) & (y2 >= 0) & (y2 < height) & (x2 >= 0)
            a = labels[y1[ok], x1[ok]].astype(np.int64)
            b = labels[y2[ok], x2[ok]].astype(np.int64)
            pair = (a > 0) & (b > 0) & (a != b)
            lo, hi = np.minimum(a[pair], b[pair]), np.maximum(a[pair], b[pair])
            # one vote per line pixel and pair: encode the pixel too, dedupe later
            keys.append(np.stack([lo, hi, np.nonzero(ok)[0][pair]], axis=1))
            flags.append(joinable[ok][pair])
    if not keys:
        return {}
    keys_all = np.concatenate(keys)
    flags_all = np.concatenate(flags)
    if len(keys_all) == 0:
        return {}
    unique, index = np.unique(keys_all, axis=0, return_index=True)
    flag = flags_all[index]
    pairs, inverse = np.unique(unique[:, :2], axis=0, return_inverse=True)
    total = np.bincount(inverse, minlength=len(pairs))
    good = np.bincount(inverse, weights=flag, minlength=len(pairs))
    return {(int(a), int(b)): (int(n), int(g)) for (a, b), n, g in zip(pairs, total, good)}


def _inscribed(labels, stats, component: int) -> float:
    x, y, w, h = stats[component, :4]
    patch = (labels[y : y + h, x : x + w] == component).astype(np.uint8)
    return float(cv2.distanceTransform(np.pad(patch, 1), cv2.DIST_L2, 3).max())


def _seed_component(labels, sizes, seed, exterior, min_pixels) -> int:
    height, width = labels.shape
    x, y = int(round(seed[0])), int(round(seed[1]))
    if not (0 <= x < width and 0 <= y < height):
        return 0
    for grow in (1, 2, 4):
        window = labels[max(0, y - 5 * grow) : y + 5 * grow + 1, max(0, x - 10 * grow) : x + 10 * grow + 1]
        values = window[window > 0]
        if len(values) == 0:
            continue
        counts = np.bincount(values)
        candidates = [c for c in np.nonzero(counts)[0] if sizes[c] > min_pixels and c not in exterior]
        if candidates:
            return int(max(candidates, key=lambda c: counts[c]))
    return 0


def _opening_pockets(labels, stats, wall, scale: float, skip: set[int], graph, exterior: set[int]) -> set[int]:
    """Free components that fill a door or window opening.

    Such a pocket is thin, has a wall end at both short ends and a drawn line along both
    long sides. At a wall end (a jamb) the wall is no thicker than the pocket; where a
    stair tread or a fixture meets a wall, the wall carries on to both sides. A window is
    drawn as two or three strips side by side: they are tested together.
    """
    height, width = labels.shape
    max_area = 3.5 * 0.6 * scale * scale
    max_thick = 0.6 * scale

    def test(components: list[int]) -> tuple[bool, bool]:
        """(is a strip between two wall ends, has a jamb)"""
        xs0 = min(stats[c, 0] for c in components)
        ys0 = min(stats[c, 1] for c in components)
        xs1 = max(stats[c, 0] + stats[c, 2] for c in components)
        ys1 = max(stats[c, 1] + stats[c, 3] for c in components)
        area = sum(stats[c, 4] for c in components)
        if area > max_area or area < 4 or min(xs1 - xs0, ys1 - ys0) > max_thick + 2:
            return False, False
        pad = 11
        x0, y0 = max(0, xs0 - pad), max(0, ys0 - pad)
        x1, y1 = min(width, xs1 + pad), min(height, ys1 + pad)
        patch = np.isin(labels[y0:y1, x0:x1], components)
        ys, xs = np.nonzero(patch)
        (cx, cy), (rw, rh), angle = cv2.minAreaRect(np.stack([xs, ys], axis=1).astype(np.float32))
        theta = np.radians(angle)
        u = np.array([np.cos(theta), np.sin(theta)])
        v = np.array([-u[1], u[0]])
        a, b = rw / 2 + 0.5, rh / 2 + 0.5
        if b > a:
            u, v, a, b = v, u, b, a
        if 2 * b > max_thick or area < 0.55 * (2 * a) * (2 * b):
            return False, False
        wy, wx = np.nonzero(wall[y0:y1, x0:x1])
        if len(wx) == 0:
            return False, False
        pu = (wx - cx) * u[0] + (wy - cy) * u[1]
        pv = (wx - cx) * v[0] + (wy - cy) * v[1]
        # a door leaf may lie between the pocket and the wall end, so look 8 px out
        zone_u = (np.abs(pu) > a) & (np.abs(pu) <= a + 8)
        ends = jambs = 0
        for sign in (-1, 1):
            end = zone_u & (np.sign(pu) == sign)
            if (end & (np.abs(pv) <= b)).sum() < 2 * (2 * b):
                continue
            ends += 1
            beyond = end & (np.abs(pv) > b + 2.5) & (np.abs(pv) <= b + 6)
            through = all((beyond & (np.sign(pv) == side)).sum() > 0.3 * 3.5 * 8 for side in (-1, 1))
            if not through:
                jambs += 1
        if ends < 2:
            return False, False
        # both long sides must be lines, not wall
        side = (np.abs(pu) < a - 2) & (np.abs(pv) > b) & (np.abs(pv) <= b + 3)
        span = max(2 * a - 4, 1) * 3
        if any((side & (np.sign(pv) == sign)).sum() > 0.4 * span for sign in (-1, 1)):
            return False, False
        return True, jambs >= 1

    strips: dict[int, bool] = {}
    for component in range(1, len(stats)):
        if component in skip:
            continue
        if stats[component, 4] > max_area or min(stats[component, 2], stats[component, 3]) > max_thick + 2:
            continue
        is_strip, jamb = test([component])
        if is_strip:
            strips[component] = jamb
    found = {c for c, jamb in strips.items() if jamb}
    # a strip between wall ends that faces the outside is a window pane
    found |= {c for c in strips if any(n in exterior for n in graph.get(c, {}))}
    # strips side by side
    seen: set[int] = set()
    for component in strips:
        if component in seen:
            continue
        group, frontier = [component], [component]
        seen.add(component)
        while frontier:
            current = frontier.pop()
            for neighbour in graph.get(current, {}):
                if neighbour in strips and neighbour not in seen:
                    seen.add(neighbour)
                    group.append(neighbour)
                    frontier.append(neighbour)
        if len(group) >= 5:
            # many strips side by side are the treads of a stair, not an opening
            found.difference_update(group)
        elif len(group) > 1:
            is_strip, jamb = test(group)
            if is_strip and jamb:
                found.update(group)
    return found


def _stair_flights(labels, stats, graph, scale: float, skip: set[int]) -> list[list[int]]:
    """Groups of four or more long narrow components side by side: the treads of a stair."""
    low, high = 0.12 * scale, 0.45 * scale
    treads: set[int] = set()
    for component in range(1, len(stats)):
        if component in skip:
            continue
        x, y, w, h, area = stats[component]
        if area < low * 2.2 * low or area > high * 3.2 * scale or min(w, h) > 3.2 * scale:
            continue
        ys, xs = np.nonzero(labels[y : y + h, x : x + w] == component)
        _, (rw, rh), _ = cv2.minAreaRect(np.stack([xs, ys], axis=1).astype(np.float32))
        short, long_ = min(rw, rh) + 1, max(rw, rh) + 1
        if low <= short <= high and long_ >= 2.2 * short and area >= 0.75 * short * long_:
            treads.add(component)
    flights = []
    seen: set[int] = set()
    for component in treads:
        if component in seen:
            continue
        group, frontier = [component], [component]
        seen.add(component)
        while frontier:
            current = frontier.pop()
            for neighbour, (pixels, good) in graph.get(current, {}).items():
                if neighbour in treads and neighbour not in seen and good >= 0.6 * pixels:
                    seen.add(neighbour)
                    group.append(neighbour)
                    frontier.append(neighbour)
        if len(group) >= 4:
            flights.append(group)
    return flights


def _refine(mask: np.ndarray, dark: np.ndarray, origin: tuple[int, int]):
    """Polygons of ``mask`` with each edge moved to the centre of the line under it.

    Returns (outer polygon, hole polygons) in image coordinates, or (None, []).
    """
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None, []
    outer_index = max(
        (i for i in range(len(contours)) if hierarchy[0][i][3] < 0),
        key=lambda i: cv2.contourArea(contours[i]),
    )
    polygons = []
    for index, contour in enumerate(contours):
        if index != outer_index and hierarchy[0][index][3] != outer_index:
            continue
        if index != outer_index and cv2.contourArea(contour) < 4:
            continue
        simple = cv2.approxPolyDP(contour, 1.3, True).reshape(-1, 2).astype(np.float64)
        if len(simple) < 3:
            continue
        polygons.append((index == outer_index, _refine_polygon(simple, mask, dark, origin)))
    outer = next((p for is_outer, p in polygons if is_outer), None)
    holes = [p for is_outer, p in polygons if not is_outer]
    return outer, holes


def _refine_polygon(points: np.ndarray, mask: np.ndarray, dark: np.ndarray, origin) -> np.ndarray:
    ox, oy = origin
    count = len(points)
    start = points
    end = np.roll(points, -1, axis=0)
    vector = end - start
    length = np.linalg.norm(vector, axis=1)
    length[length == 0] = 1e-9
    direction = vector / length[:, None]
    normal = np.stack([direction[:, 1], -direction[:, 0]], axis=1)
    # Point the normal out of the region.
    middle = (start + end) / 2
    probe = np.round(middle + 1.5 * normal).astype(int)
    inside = np.zeros(count, bool)
    ok = (probe[:, 0] >= 0) & (probe[:, 0] < mask.shape[1]) & (probe[:, 1] >= 0) & (probe[:, 1] < mask.shape[0])
    inside[ok] = mask[probe[ok, 1], probe[ok, 0]] > 0
    normal[inside] *= -1

    steps = np.arange(-2.5, 6.01, 0.25)
    offsets = np.full(count, np.nan)
    sample_edge, sample_point = [], []
    for i in range(count):
        if length[i] < 5:
            continue
        along = np.arange(2.0, length[i] - 1.99, 1.0)
        if len(along) == 0:
            continue
        sample_edge.append(np.full(len(along), i))
        sample_point.append(start[i] + along[:, None] * direction[i])
    if sample_edge:
        edge = np.concatenate(sample_edge)
        point = np.concatenate(sample_point)
        xs = point[:, 0, None] + steps[None, :] * normal[edge, 0, None] + ox
        ys = point[:, 1, None] + steps[None, :] * normal[edge, 1, None] + oy
        profile = ndimage.map_coordinates(dark, [ys, xs], order=1, mode="nearest")
        centre = _line_centre(profile, steps)
        for i in np.unique(edge):
            values = centre[edge == i]
            values = values[~np.isnan(values)]
            if len(values) >= max(2, 0.3 * (edge == i).sum()):
                offsets[i] = np.median(values)
    known = ~np.isnan(offsets)
    default = float(np.average(offsets[known], weights=length[known])) if known.any() else 1.4
    offsets[~known] = default
    offsets = np.clip(offsets, -1.0, 5.0)

    # Move each edge along its normal, then intersect neighbouring edges.
    base = start + offsets[:, None] * normal
    moved = np.empty_like(points)
    for i in range(count):
        j = (i - 1) % count
        d1, d2 = direction[j], direction[i]
        cross = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(cross) < 0.2:
            moved[i] = (base[j] + vector[j] + base[i]) / 2
            continue
        t = ((base[i][0] - base[j][0]) * d2[1] - (base[i][1] - base[j][1]) * d2[0]) / cross
        candidate = base[j] + t * d1
        if np.linalg.norm(candidate - points[i]) > 8:
            candidate = (base[j] + vector[j] + base[i]) / 2
        moved[i] = candidate
    return moved + [ox, oy]


def _line_centre(profile: np.ndarray, steps: np.ndarray) -> np.ndarray:
    """Position of the centre of the first line met along each profile, or NaN."""
    rows = np.arange(len(profile))
    step = steps[1] - steps[0]
    hit = profile >= 40
    first = hit.argmax(axis=1)
    found = hit.any(axis=1) & (first > 0)
    span = int(round(2.0 / step))
    index = np.minimum(first[:, None] + np.arange(span + 1)[None, :], profile.shape[1] - 1)
    peak = np.take_along_axis(profile, index, axis=1).max(axis=1)
    half = np.where(peak >= 200, 127.5, peak / 2)
    # first crossing of the half level
    over = profile >= half[:, None]
    k = over.argmax(axis=1)
    found &= over.any(axis=1) & (k > 0)
    k = np.maximum(k, 1)
    before, after = profile[rows, k - 1], profile[rows, k]
    crossing = steps[k - 1] + step * (half - before) / np.maximum(after - before, 1e-6)

    black = peak >= 200
    centre = crossing + np.where(black, HALF_WIDTH_BLACK, HALF_WIDTH_THIN)
    # A wall outline has the grey wall fill behind it. A fixture drawn against the wall
    # thickens the outline on the room side only, so measure from the fill side.
    count = profile.shape[1]
    columns = np.arange(count)[None, :]
    after = (columns >= k[:, None]) & (profile < FILL_EDGE)
    j = after.argmax(axis=1)
    has_end = after.any(axis=1) & (j > 0)
    j = np.maximum(j, 1)
    outer = steps[j - 1] + step * (profile[rows, j - 1] - FILL_EDGE) / np.maximum(
        profile[rows, j - 1] - profile[rows, j], 1e-6
    )
    behind = np.take_along_axis(profile, np.minimum(j[:, None] + np.arange(2, 6)[None, :], count - 1), axis=1)
    fill_behind = (behind.mean(axis=1) >= 50) & (behind.mean(axis=1) <= 110) & (j + 5 < count)
    width = outer - crossing
    use_outer = black & has_end & fill_behind & (width >= 1.4) & (width <= 3.8)
    centre = np.where(use_outer, outer - (OUTLINE_WIDTH - HALF_WIDTH_BLACK), centre)
    # A thin line with empty space behind it: its centre is the centroid of its darkness.
    lo = np.maximum(first - int(round(1.25 / step)), 0)
    window = np.arange(int(round(4.0 / step)) + 1)
    index = np.minimum(lo[:, None] + window[None, :], profile.shape[1] - 1)
    values = np.take_along_axis(profile, index, axis=1)
    position = steps[index]
    isolated = ~black & (values[:, -1] < 12) & (values[:, 0] < 12)
    centroid = (values * position).sum(axis=1) / np.maximum(values.sum(axis=1), 1e-6)
    centre = np.where(isolated, centroid, centre)
    centre[~found] = np.nan
    return centre


def _shoelace(points: np.ndarray) -> float:
    x, y = points[:, 0], points[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def measure_rooms(
    image: np.ndarray,
    seeds: list[tuple[float, float]],
    pixels_per_metre: float,
    text_boxes: list[tuple[float, float, float, float]] = (),
    raster: Raster | None = None,
) -> list[RoomMeasure]:
    """Measure the room around each seed point. ``image`` is an RGB array."""
    raster = raster or classify(image, text_boxes, pixels_per_metre)
    height, width = raster.grey.shape
    free = (~raster.ink).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(free, connectivity=4)
    sizes = stats[:, cv2.CC_STAT_AREA]
    border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    exterior = set(border.tolist()) - {0}
    exterior.add(int(np.argmax(sizes[1:]) + 1))

    contacts = _contacts(labels, raster)
    graph: dict[int, dict[int, tuple[int, int]]] = {}
    for (a, b), value in contacts.items():
        graph.setdefault(a, {})[b] = value
        graph.setdefault(b, {})[a] = value

    scale = pixels_per_metre
    rooms = [RoomMeasure(seed=(float(x), float(y))) for x, y in seeds]
    seed_of = [_seed_component(labels, sizes, seed, exterior, 0.03 * scale * scale) for seed in seeds]
    owner: dict[int, int] = {}
    for index, component in enumerate(seed_of):
        if component == 0:
            rooms[index].flags.append("no-region")
        elif component in owner:
            rooms[index].flags.append("shared-region")
            rooms[owner[component]].flags.append("shared-region")
        else:
            owner[component] = index

    # The pocket of a door or window opening lies in the wall between two rooms: it belongs
    # to neither, and nothing may join a room through it.
    neutral = _opening_pockets(labels, stats, raster.wall, scale, set(owner) | exterior, graph, exterior)
    in_gap = np.bincount(labels[raster.opening], minlength=count)
    neutral |= {
        int(c) for c in np.nonzero(in_gap >= 0.7 * np.maximum(sizes, 1))[0] if c and c not in owner and c not in exterior
    }

    # A stair flight belongs as a whole to the room whose label sits on it.
    for flight in _stair_flights(labels, stats, graph, scale, exterior):
        neutral.difference_update(flight)
        rooms_here = {owner[c] for c in flight if c in owner}
        if len(rooms_here) == 1:
            room = rooms_here.pop()
            for component in flight:
                owner[component] = room

    # A shaded floor (an outdoor room, a stairwell) ends where the shade ends.
    # Only flat pixels count: next to a line the grey is a blend.
    flat = (cv2.dilate(raster.grey, K3) - cv2.erode(raster.grey, K3)) <= 1
    flat_count = np.bincount(labels[flat], minlength=count)
    shade = np.bincount(labels[flat], weights=raster.grey[flat], minlength=count) / np.maximum(flat_count, 1)
    shade[flat_count < 12] = np.nan
    room_shade = {index: shade[component] for component, index in owner.items() if component == seed_of[index]}

    def joins(pixels: int, good: int) -> bool:
        return pixels >= 3 and good >= 0.6 * pixels

    changed = True
    while changed:
        changed = False
        candidates = set()
        for component in owner:
            candidates.update(graph.get(component, {}))
        new: dict[int, int] = {}
        for component in candidates:
            if component in owner or component in exterior or component in neutral:
                continue
            votes: dict[int, int] = {}
            outside = 0
            for neighbour, (pixels, good) in graph[component].items():
                if not joins(pixels, good) or neighbour in neutral:
                    continue
                if neighbour in exterior:
                    outside += pixels
                elif neighbour in owner:
                    votes[owner[neighbour]] = votes.get(owner[neighbour], 0) + pixels
            if not votes or outside:
                continue
            room, weight = max(votes.items(), key=lambda item: item[1])
            # (stair treads are white on a lightly shaded stairwell, so only the darker
            # outdoor shade counts; the comparison is False when a shade is unknown)
            if (shade[component] < OUTDOOR_SHADE_MAX) != (room_shade.get(room, np.nan) < OUTDOOR_SHADE_MAX) and not (
                np.isnan(shade[component]) or np.isnan(room_shade.get(room, np.nan))
            ):
                continue
            if weight >= 0.85 * sum(votes.values()):
                new[component] = room
        if new:
            owner.update(new)
            changed = True

    members: dict[int, list[int]] = {}
    for component, index in owner.items():
        members.setdefault(index, []).append(component)

    fill_region = cv2.morphologyEx(
        ((np.abs(raster.grey - WALL_GREY) <= 8) & ~raster.red).astype(np.uint8), cv2.MORPH_OPEN, K3
    ).astype(bool)
    for index, room in enumerate(rooms):
        components = members.get(index)
        if not components:
            if seed_of[index] and "shared-region" in room.flags:
                components = [seed_of[index]]
            else:
                continue
        x0 = max(0, min(stats[c, 0] for c in components) - 8)
        y0 = max(0, min(stats[c, 1] for c in components) - 8)
        x1 = min(width, max(stats[c, 0] + stats[c, 2] for c in components) + 8)
        y1 = min(height, max(stats[c, 1] + stats[c, 3] for c in components) + 8)
        sub = labels[y0:y1, x0:x1]
        region = np.isin(sub, components)
        boundary = raster.wall[y0:y1, x0:x1] | (raster.line[y0:y1, x0:x1] & ~raster.joinable[y0:y1, x0:x1])
        closed = region | (cv2.morphologyEx(region.astype(np.uint8), cv2.MORPH_CLOSE, K5).astype(bool) & ~boundary)
        closed |= cv2.morphologyEx(closed.astype(np.uint8), cv2.MORPH_CLOSE, K3).astype(bool) & ~raster.wall[y0:y1, x0:x1]
        # Fill what lies inside the room, but keep a hole that holds wall (a column, a shaft).
        holes = ndimage.binary_fill_holes(closed) & ~closed
        hole_labels, hole_count = ndimage.label(holes)
        if hole_count:
            has_wall = ndimage.maximum(fill_region[y0:y1, x0:x1], hole_labels, np.arange(1, hole_count + 1))
            keep = np.zeros(hole_count + 1, bool)
            keep[1:] = np.asarray(has_wall, bool)
            closed |= holes & ~keep[hole_labels]
        room.box = (x0, y0, x1, y1)
        room.pixel_area_m2 = float(closed.sum() / scale**2)
        floor = raster.grey[y0:y1, x0:x1][region]
        room.shade = float(np.median(floor)) if len(floor) else None
        if closed[0].any() or closed[-1].any() or closed[:, 0].any() or closed[:, -1].any():
            room.flags.append("open-region")
        outer, hole_polygons = _refine(closed, raster.dark, (x0, y0))
        if outer is None:
            room.flags.append("no-region")
            continue
        area = _shoelace(outer) - sum(_shoelace(h) for h in hole_polygons)
        room.area_m2 = float(area / scale**2)
        room.polygon = [(float(x), float(y)) for x, y in outer]
        room.holes = [[(float(x), float(y)) for x, y in h] for h in hole_polygons]
    return rooms
