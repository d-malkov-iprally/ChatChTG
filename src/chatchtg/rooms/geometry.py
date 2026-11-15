"""Counts from shapes, without reading text: storeys on a Finnish sheet, apartments on a Swiss one.

Ported from the research prototype. Images are OpenCV BGR arrays.
"""

from __future__ import annotations

import cv2
import numpy as np


def decode(content: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Not a readable image")
    return image


def count_storeys(
    image: np.ndarray,
    dilate_fraction: float = 0.012,
    relative_area: float = 0.2,
    hollow: float = 1.6,
    threshold: int = 150,
) -> tuple[int, list[tuple[int, int, int, int]]]:
    """Count the separate floor drawings on a sheet. Return the count and their boxes (x, y, w, h).

    Ink is closed into blobs. A floor drawing is a large blob that is mostly empty inside;
    a title block or a text paragraph is dense.
    """
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = grey.shape
    diagonal = (height * height + width * width) ** 0.5
    ink = (grey < threshold).astype(np.uint8)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    small = stats[:, 4] < max(20, (diagonal * 0.01) ** 2 * 0.1)
    small[0] = False
    ink[small[labels]] = 0
    radius = max(3, int(dilate_fraction * diagonal))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    closed = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 0.01 * height * width:
            continue
        mask = np.zeros_like(closed)
        cv2.drawContours(mask, [contour], -1, 1, -1)
        inked = int((ink & mask).sum())
        blobs.append((area, area / max(1, inked), cv2.boundingRect(contour)))
    if not blobs:
        return 1, []
    hollow_areas = [area for area, ratio, _ in blobs if ratio >= hollow]
    largest = max(hollow_areas) if hollow_areas else max(area for area, _, _ in blobs)
    kept = [
        box for area, ratio, box in blobs if area >= relative_area * largest and ratio >= hollow
    ]
    return max(1, len(kept)), [tuple(int(v) for v in box) for box in kept]


def _red_mask(image: np.ndarray) -> np.ndarray:
    blue, green, red = (image[..., i].astype(np.int16) for i in range(3))
    return ((red - green > 50) & (red - blue > 40)).astype(np.uint8) * 255


def _wall_mask(image: np.ndarray) -> np.ndarray:
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.int16)
    saturation = image.max(2).astype(np.int16) - image.min(2)
    return ((grey >= 150) & (grey <= 200) & (saturation < 20)).astype(np.uint8)


def count_apartments(
    image: np.ndarray, bridge: int = 70, min_width: float = 12, min_area: int = 20000, band: int = 10
) -> tuple[int, list[tuple[int, int, int, int]]]:
    """Count the areas enclosed by the red unit outlines of a Swiss plan.

    Return the count and the boxes (x, y, w, h) of the units. The outline is dash-dotted and
    open at the entrance door, so it is closed along both axes first. A stairwell cut out of
    a unit is also enclosed; it is dropped because its walls lie inside the red line, not outside.
    """
    red = cv2.dilate(_red_mask(image), np.ones((3, 3), np.uint8))
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    across = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((1, bridge), np.uint8))
    down = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((bridge, 1), np.uint8))
    outline = red | across | down
    free = (outline == 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(free, connectivity=4)
    distance = cv2.distanceTransform(free, cv2.DIST_L2, 3)
    widest = np.zeros(count)
    np.maximum.at(widest, labels.ravel(), distance.ravel())
    height, width = red.shape
    walls = _wall_mask(image)
    kernel = np.ones((3, 3), np.uint8)
    units = []
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        if x == 0 or y == 0 or x + w == width or y + h == height:
            continue
        if area < min_area or widest[i] < min_width:
            continue
        component = (labels == i).astype(np.uint8)
        inner = component & (1 - cv2.erode(component, kernel, iterations=band))
        grown = cv2.dilate(component, kernel, iterations=band + 12)
        outer = grown & (1 - component) & (outline == 0)
        wall_inside = walls[inner > 0].mean() if inner.any() else 0
        wall_outside = walls[outer > 0].mean() if outer.any() else 0
        if wall_outside >= wall_inside:
            units.append((int(x), int(y), int(w), int(h)))
    return len(units), units
