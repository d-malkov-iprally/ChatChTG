"""Which apartment a room belongs to, and whether it is outdoor, read from the image.

Apartment: every apartment is drawn with a red dash-dot outline and carries a boxed tag
("Whg. A", "App. B"). The red outline, thickened until its gaps close, cuts the image into
regions. The region that holds a tag is that apartment, and a room belongs to the apartment
whose region holds its label.

Outdoor: the floor of an outdoor room (balcony, loggia, terrace, ...) is shaded grey, the
floor of an indoor room is white. The shading is the only reliable sign: many outdoor
rooms have no name.
"""

from __future__ import annotations

from collections import Counter

import cv2
import numpy as np

Point = tuple[float, float]

OUTLINE_RADII = (3, 4, 5, 6, 8, 10)
"""How far the red outline is thickened, in pixels. The smallest radius that gives every
tag a closed region of its own is used: a larger one closes more gaps but can also swallow
a narrow room."""
SEED_REACH = 12
"""A point that lies on the thickened outline counts for the nearest region this near."""
TAG_REACH = 80
"""A tag outside every closed region is moved to the nearest free one this near."""

TAG_BOX = (40, 16)
"""Half the width and height of the box around an apartment tag, generously."""

OUTDOOR_GREY = (224, 232)
"""The shading of an outdoor floor is grey 228."""
LIGHT = 200
"""Brighter than this is floor; darker is a line or a glyph."""
OUTDOOR_WINDOW = 18
OUTDOOR_SAMPLE = 150
"""The room is judged by this many light pixels, the ones nearest to its label."""
OUTDOOR_SHARE = 0.5


def red_outline(image: np.ndarray) -> np.ndarray:
    """Mask of the red apartment outline, about RGB (176, 58, 46)."""
    r, g, b = (image[..., i].astype(np.int16) for i in range(3))
    return ((r > 120) & (r - g > 60) & (r - b > 60)).astype(np.uint8)


def _regions(red: np.ndarray, radius: int) -> tuple[np.ndarray, set[int]]:
    """Label the space left by the thickened outline. 0 is the outline itself.

    Also returns the labels that touch the image border: they are outside every apartment.
    """
    size = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    free = 1 - cv2.dilate(red, kernel)
    _, labels = cv2.connectedComponents(free, connectivity=4)
    edge = np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    return labels, set(np.unique(edge).tolist()) - {0}


def _nearest_regions(labels: np.ndarray, point: Point, reach: int) -> list[int]:
    """Labels of the regions within ``reach`` pixels of a point, nearest first."""
    height, width = labels.shape
    x, y = int(round(point[0])), int(round(point[1]))
    if not (-reach <= x < width + reach and -reach <= y < height + reach):
        return []
    x0, x1 = max(x - reach, 0), min(x + reach + 1, width)
    y0, y1 = max(y - reach, 0), min(y + reach + 1, height)
    window = labels[y0:y1, x0:x1]
    ys, xs = np.nonzero(window)
    if len(ys) == 0:
        return []
    distance = (ys + y0 - y) ** 2 + (xs + x0 - x) ** 2
    order = np.argsort(distance, kind="stable")
    order = order[distance[order] <= reach * reach]
    found: list[int] = []
    for label in window[ys[order], xs[order]].tolist():
        if label not in found:
            found.append(label)
    return found


def _region_at(labels: np.ndarray, point: Point) -> int:
    near = _nearest_regions(labels, point, SEED_REACH)
    return near[0] if near else 0


def _tag_regions(
    labels: np.ndarray, outside: set[int], tags: dict[str, Point]
) -> dict[str, int]:
    """The tags that sit in a closed region of their own, with that region."""
    at = {letter: _region_at(labels, point) for letter, point in tags.items()}
    shared = Counter(at.values())
    return {
        letter: region
        for letter, region in at.items()
        if region and region not in outside and shared[region] == 1
    }


def _closed_regions(
    red: np.ndarray, tags: dict[str, Point]
) -> tuple[np.ndarray, set[int], dict[str, int]]:
    """Cut the image at the outline, thickened as little as gives most tags a region."""
    best: tuple[np.ndarray, set[int], dict[str, int]] | None = None
    for radius in OUTLINE_RADII:
        labels, outside = _regions(red, radius)
        owned = _tag_regions(labels, outside, tags)
        if best is None or len(owned) > len(best[2]):
            best = (labels, outside, owned)
        if len(owned) == len(tags):
            break
    assert best is not None
    return best


def assign_apartments(
    image: np.ndarray, room_points: list[Point], tags: dict[str, Point]
) -> list[str | None]:
    """The apartment letter of every room, or None for a room outside every apartment.

    ``image`` is RGB. ``room_points`` are the positions of the room labels and ``tags`` the
    positions of the apartment tags, in pixels (x, y).
    """
    if not tags or not room_points:
        return [None] * len(room_points)
    red = red_outline(image)
    labels, outside, owned = _closed_regions(red, tags)
    if len(owned) < len(tags):
        # A tag box drawn on top of an outline breaks it. Fill the box in and try again.
        for letter, (x, y) in tags.items():
            if letter not in owned:
                half_width, half_height = TAG_BOX
                corner = (int(x) - half_width, int(y) - half_height)
                cv2.rectangle(red, corner, (int(x) + half_width, int(y) + half_height), 1, -1)
        labels, outside, owned = _closed_regions(red, tags)
    # A tag that sits outside its outline, or in the region of another tag, takes the
    # nearest closed region that no tag owns.
    taken = set(owned.values())
    for letter, point in tags.items():
        if letter in owned:
            continue
        for region in _nearest_regions(labels, point, TAG_REACH):
            if region not in outside and region not in taken:
                owned[letter] = region
                taken.add(region)
                break
    letter_of = {region: letter for letter, region in owned.items()}
    # A loop without a tag stays without an apartment. Joining it to the tagged loop next
    # to it was tried and is wrong far more often than right: such a loop is mostly a
    # unit of its own, or a room the reference answers leave out.
    return [letter_of.get(_region_at(labels, point)) for point in room_points]

def outdoor_rooms(image: np.ndarray, room_points: list[Point]) -> list[bool]:
    """Whether each room is outdoor: its floor around the label is shaded grey.

    Looks at the light pixels nearest to the label, so the glyphs of the label and the
    lines of the room do not count, and a small room is not judged by its neighbours.
    """
    height, width = image.shape[:2]
    low, high = OUTDOOR_GREY
    result = []
    for px, py in room_points:
        x, y = int(round(px)), int(round(py))
        x0, y0 = max(x - OUTDOOR_WINDOW, 0), max(y - OUTDOOR_WINDOW, 0)
        window = image[
            y0 : max(min(y + OUTDOOR_WINDOW + 1, height), 0),
            x0 : max(min(x + OUTDOOR_WINDOW + 1, width), 0),
        ].astype(np.int16)
        r, g, b = window[..., 0], window[..., 1], window[..., 2]
        ys, xs = np.nonzero((np.abs(r - g) < 6) & (np.abs(r - b) < 6) & (r > LIGHT))
        if len(ys) == 0:
            result.append(False)
            continue
        nearest = np.argsort((ys + y0 - y) ** 2 + (xs + x0 - x) ** 2, kind="stable")
        floor = r[ys[nearest[:OUTDOOR_SAMPLE]], xs[nearest[:OUTDOOR_SAMPLE]]]
        result.append(bool(((floor >= low) & (floor <= high)).mean() > OUTDOOR_SHARE))
    return result
