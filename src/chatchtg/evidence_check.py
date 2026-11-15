"""Checks the positions that an LLM gave for the items it counted, and makes evidence items.

The check is about consistency only: it compares the positions with the image size, with
each other and with the count that the answer states. It does not look at the pixels, so
it cannot tell whether a mark is on an item of the asked kind.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from chatchtg.llm.client import Location

LLM_LOCATED = "llm-located"
"""``AnswerResponse.method`` of an answer whose evidence is the marks of the LLM."""
LLM_KIND = "llm"
"""The ``kind`` of such an evidence item (see chatchtg.evidence)."""

# Two marks closer than this may show the same item: this part of the longer image side,
# and at least this many pixels (which is also the distance when the size is unknown).
DUPLICATE_PART = 0.01
DUPLICATE_PIXELS = 6


@dataclass(frozen=True)
class CheckedEvidence:
    evidence: list[dict[str, Any]]
    """Items ``{"text": "<label> <n>", "kind": "llm", "point": [x, y]}`` in whole pixels.
    ``n`` is the 1-based position of the item in the model's list."""
    note: str | None
    """What the check found, in sentences for the user. None when it found nothing."""


def check_locations(
    locations: list[Location], image_size: tuple[int, int] | None, stated: float | None
) -> CheckedEvidence:
    """Turn the model's ``locations`` into evidence items and report what does not fit.

    ``image_size`` is ``(width, height)`` of the drawing, None when it is unknown.
    ``stated`` is the count that the answer text states, None when no number was read.
    A position outside the image is dropped. Positions that are very close to each other
    and a stated count that differs from the number of listed positions are reported, and
    the items are kept: the text and the marks are both what the model said.
    """
    near = float(DUPLICATE_PIXELS)
    if image_size is not None:
        near = max(near, DUPLICATE_PART * max(image_size))
    evidence: list[dict[str, Any]] = []
    kept: list[tuple[float, float]] = []
    dropped = duplicates = 0
    for number, location in enumerate(locations, start=1):
        x, y = location.point
        if not (math.isfinite(x) and math.isfinite(y)):
            dropped += 1
            continue
        if image_size is not None and not (0 <= x < image_size[0] and 0 <= y < image_size[1]):
            dropped += 1
            continue
        if any(math.hypot(x - other_x, y - other_y) < near for other_x, other_y in kept):
            duplicates += 1
        kept.append((x, y))
        evidence.append(
            {"text": f"{location.label} {number}", "kind": LLM_KIND, "point": [round(x), round(y)]}
        )
    sentences = []
    if stated is not None and stated != len(locations):
        sentences.append(f"The answer says {stated:g}, but {len(locations)} positions are marked.")
    if dropped:
        sentences.append(f"{dropped} marked positions were outside the drawing and are not shown.")
    if duplicates:
        sentences.append(
            f"{duplicates} marks are very close to another mark and may show the same item."
        )
    return CheckedEvidence(evidence=evidence, note=" ".join(sentences) or None)
