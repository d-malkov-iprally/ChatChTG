"""The object counter that the answer service uses: it counts stored detections."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from chatchtg.extractions.models import Extraction, Item
from chatchtg.extractions.store import ExtractionStore
from chatchtg.objects.questions import ObjectQuestion, classify

PRODUCER = "rfdetr-large-1024"
"""The producer name of the detections of the ``large-1024`` run of ``object_detector``."""
PRODUCER_VERSION = 1

Thresholds = Mapping[str, float]
"""Detector class -> the minimum score of a detection that is counted."""

# Chosen on the valid split of CubiCasa5K only, to give the most exact counts there
# (`python -m chatchtg.objects.tune`, docs/benchmarks/2026-11-15-object-counter.md).
THRESHOLDS: dict[str, float] = {
    "window": 0.46,
    "door": 0.53,
    "toilet": 0.48,
    "sink": 0.34,
    "shower": 0.35,
    # Bathtubs are rare (38 on the valid plans) and most bathtub detections there were wrong,
    # so nearly only the surest are counted.
    "bathtub": 0.9,
}


@dataclass
class ObjectFinding:
    question: ObjectQuestion
    count: int
    method: str
    """How the count was made: ``detector``."""
    evidence: list[dict[str, Any]] = field(default_factory=list)
    """The counted objects: ``{"text", "box", "ref"}``, the box in image pixels (x0, y0, x1, y1)."""


def counted(extraction: Extraction, classes: tuple[str, ...], thresholds: Thresholds) -> list[Item]:
    """The detected objects of ``classes`` whose score reaches the minimum of their class."""
    return [
        item
        for item in extraction.items
        if item.kind == "object"
        and item.label in classes
        and (item.score if item.score is not None else 1.0) >= thresholds[item.label]
    ]


def count(
    extraction: Extraction, question: ObjectQuestion, thresholds: Thresholds = THRESHOLDS
) -> ObjectFinding:
    """Answer ``question`` from the detections of one drawing."""
    items = counted(extraction, question.classes, thresholds)
    evidence = [
        {
            "text": f"{item.label} {item.score:.2f}" if item.score is not None else item.label,
            "box": [round(v) for v in item.box],
            "ref": str(extraction.ref(item.id)),
        }
        for item in items
    ]
    return ObjectFinding(question, len(items), "detector", evidence)


class DetectorObjectCounter:
    """Counts objects from the detections stored for a drawing.

    A drawing without stored detections of ``producer`` gets None, and the LLM answers.
    """

    def __init__(
        self,
        store: ExtractionStore,
        producer: str = PRODUCER,
        version: int | None = PRODUCER_VERSION,
        thresholds: Thresholds = THRESHOLDS,
    ) -> None:
        self._store = store
        self._producer = producer
        self._version = version
        self._thresholds = thresholds

    def solve(self, document_id: str, message: str) -> ObjectFinding | None:
        question = classify(message)
        if question is None:
            return None
        try:
            extraction = self._store.get(document_id, self._producer, self._version)
        except Exception:  # a detections file that cannot be read must not lose the answer
            logging.getLogger(__name__).warning(
                "The detections of %s cannot be read; the LLM answers", document_id, exc_info=True
            )
            return None
        if extraction is None:
            return None
        return count(extraction, question, self._thresholds)
