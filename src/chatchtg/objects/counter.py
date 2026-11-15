"""The object counter that the answer service uses: it counts stored detections."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from chatchtg.extractions.models import Extraction, Item
from chatchtg.extractions.store import ExtractionStore
from chatchtg.objects.questions import ObjectQuestion, classify

PRODUCER = "rfdetr-large-1024"
"""The producer name of the detections of the RF-DETR detector of ``object_detector``.
Version 1 is the run ``large-1024``. The name stayed when later runs became new versions."""

Thresholds = Mapping[str, float]
"""Detector class -> the minimum score of a detection that is counted."""

# Detector class -> minimum score, for each producer version (each is another model).
# Chosen on the valid split of CubiCasa5K only, to give the most exact counts there
# (`python -m chatchtg.objects.tune`, docs/benchmarks/2026-11-15-object-counter.md).
THRESHOLDS_BY_VERSION: dict[int, dict[str, float]] = {
    # Run large-1024.
    1: {
        "window": 0.46,
        "door": 0.53,
        "toilet": 0.48,
        "sink": 0.34,
        "shower": 0.35,
        # Bathtubs are rare (38 on the valid plans) and most bathtub detections there were
        # wrong, so nearly only the surest are counted.
        "bathtub": 0.9,
    },
    # Run large-1536-ft: large-1024 trained 10 more epochs at 1536 px. Not used by the app:
    # it was no better on the test split (docs/benchmarks/2026-11-15-object-counter.md).
    # Kept so that `report --version 2` can repeat that measurement.
    2: {
        "window": 0.46,
        "door": 0.45,
        "toilet": 0.47,
        "sink": 0.34,
        "shower": 0.25,
        "bathtub": 0.95,
    },
}
PRODUCER_VERSION = 1
"""The version that the app reads."""
THRESHOLDS = THRESHOLDS_BY_VERSION[PRODUCER_VERSION]


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
    extraction: Extraction, question: ObjectQuestion, thresholds: Thresholds | None = None
) -> ObjectFinding:
    """Answer ``question`` from the detections of one drawing. Without ``thresholds``, the
    minimum scores of the extraction's producer version."""
    if thresholds is None:
        thresholds = THRESHOLDS_BY_VERSION[extraction.version]
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
        thresholds: Thresholds | None = None,
    ) -> None:
        self._store = store
        self._producer = producer
        self._version = version
        self._thresholds = thresholds

    def solve(self, document_id: str, message: str) -> ObjectFinding | None:
        question = classify(message)
        if question is None:
            return None
        extraction = self._store.get(document_id, self._producer, self._version)
        if extraction is None:
            return None
        return count(extraction, question, self._thresholds)
