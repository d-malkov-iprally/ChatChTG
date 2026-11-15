"""The object counter that the answer service uses: it counts stored detections.

There is one counter and one profile for each symbol detector. A profile says which stored
detections it reads (producer and version), which detector classes each kind of question
counts, and the minimum score of each class.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from chatchtg.extractions.models import Extraction, Item
from chatchtg.extractions.store import ExtractionStore
from chatchtg.objects.questions import ObjectQuestion, classify

Thresholds = Mapping[str, float]
"""Detector class -> the minimum score of a detection that is counted."""


@dataclass(frozen=True)
class DetectorProfile:
    name: str
    """Short name, used in settings and reports: ``cubicasa`` or ``swiss``."""
    producer: str
    """The producer name of the stored detections."""
    version: int
    """The producer version that the app reads."""
    classes: Mapping[str, tuple[str, ...]]
    """Question kind -> the detector classes whose objects it counts together. A kind that is
    not here is not answered from this detector."""
    thresholds_by_version: Mapping[int, Thresholds]
    """Producer version (each is another training run) -> the minimum score of each class."""
    runs_by_version: Mapping[int, str]
    """Producer version -> the training run whose detections it holds."""
    source_folder: str
    """The folder of its plans in the paths of the benchmark's questions."""

    @property
    def thresholds(self) -> Thresholds:
        return self.thresholds_by_version[self.version]


# The detector of `object_detector/` trained on CubiCasa5K (Finnish plans).
PRODUCER = "rfdetr-large-1024"
"""The producer name of the detections of the CubiCasa5K detector. Version 1 is the run
``large-1024``. The name stayed when later runs became new versions."""

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
RUN_BY_VERSION = {1: "large-1024", 2: "large-1536-ft"}
"""The training run whose detections each producer version holds."""
PRODUCER_VERSION = 1
"""The version that the app reads."""
THRESHOLDS = THRESHOLDS_BY_VERSION[PRODUCER_VERSION]

CUBICASA = DetectorProfile(
    name="cubicasa",
    producer=PRODUCER,
    version=PRODUCER_VERSION,
    classes={
        "window": ("window",),
        "door": ("door",),
        "toilet": ("toilet",),
        "sink": ("sink",),
        "bath_shower": ("shower", "bathtub"),
    },
    thresholds_by_version=THRESHOLDS_BY_VERSION,
    runs_by_version=RUN_BY_VERSION,
    source_folder="data/cubicasa5k",
)

# The Swiss Dwellings detector (`object_detector/modal_swiss.py`, run `swiss-medium-1280`,
# docs/benchmarks/2026-11-15-swiss-symbol-detector.md). Its minimum scores were chosen on the
# 150 valid plans by `modal_swiss::evaluate` (grid 0.10 to 0.95, step 0.05) and are written
# in its detections file; the importer refuses a file whose minimums differ from these.
SWISS = DetectorProfile(
    name="swiss",
    producer="rfdetr-swiss",
    version=1,
    classes={
        "window": ("window",),
        "door": ("door",),
        "toilet": ("toilet",),
        "washbasin": ("washbasin",),
        "bathtub": ("bathtub",),
        "shower": ("shower",),
    },
    thresholds_by_version={
        1: {
            "door": 0.55,
            "window": 0.40,
            "toilet": 0.50,
            "washbasin": 0.70,
            "bathtub": 0.30,
            "shower": 0.80,
        },
    },
    runs_by_version={1: "swiss-medium-1280"},
    source_folder="data/swiss_dwellings",
)

PROFILES: tuple[DetectorProfile, ...] = (CUBICASA, SWISS)

DEFAULT_SKIP: frozenset[tuple[str, str]] = frozenset((SWISS.name, kind) for kind in SWISS.classes)
"""(profile, kind) pairs that the counter leaves to the LLM unless
``CHATCHTG_OBJECT_COUNTER_SKIP`` says otherwise: every Swiss kind.

The Swiss detector was trained on labels from the DXF twins of the plans. Whether a model
trained on them may answer is not decided: no ADR covers it (object_detector/README.md). On
its 727 training plans it has learnt from the DXF of the very plan that is asked about. So
the Swiss kinds are answered only when the variable is set: empty for all six, or with the
kinds to leave out.

Doors (79.1% exact on the test plans) and windows (74.2%) need a decision of their own. Their
benchmark answers count doors and windows that the DXF holds twice at one place and the PNG
shows once, and the counter matches a door answer by counting boxes that are not visible
doors. No LLM has been measured on held-out Swiss plans.
(docs/benchmarks/2026-11-15-swiss-object-counter.md)"""


def profile_of(producer: str) -> DetectorProfile:
    for profile in PROFILES:
        if profile.producer == producer:
            return profile
    raise ValueError(f"No detector profile reads the producer {producer!r}")


def parse_skip(text: str) -> frozenset[tuple[str, str]]:
    """``"swiss:door, swiss:window"`` -> the (profile, kind) pairs. Empty text: none."""
    pairs = set()
    for entry in text.split(","):
        entry = entry.strip()
        if not entry:
            continue
        name, _, kind = entry.partition(":")
        profile = next((p for p in PROFILES if p.name == name.strip()), None)
        if profile is None or kind.strip() not in profile.classes:
            known = ", ".join(f"{p.name}:{k}" for p in PROFILES for k in p.classes)
            raise ValueError(f"Unknown object counter kind {entry!r}. Available: {known}")
        pairs.add((profile.name, kind.strip()))
    return frozenset(pairs)


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
    """Answer ``question`` from the detections of one drawing. The classes are those of the
    extraction's detector; without ``thresholds``, the minimum scores of its producer version."""
    profile = profile_of(extraction.producer)
    classes = profile.classes.get(question.kind)
    if classes is None:
        raise ValueError(f"{profile.name} detections do not answer {question.kind!r} questions")
    if thresholds is None:
        if extraction.version not in profile.thresholds_by_version:
            raise ValueError(
                f"No minimum scores for {extraction.producer}@{extraction.version}: "
                "add them to its profile's thresholds_by_version"
            )
        thresholds = profile.thresholds_by_version[extraction.version]
    items = counted(extraction, classes, thresholds)
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

    The first profile that answers the question's kind and has stored detections of the
    drawing counts. A drawing without stored detections, or a (profile, kind) in ``skip``,
    gets None, and the LLM answers.
    """

    def __init__(
        self,
        store: ExtractionStore,
        profiles: Iterable[DetectorProfile] = PROFILES,
        thresholds: Thresholds | None = None,
        skip: Iterable[tuple[str, str]] = DEFAULT_SKIP,
    ) -> None:
        self._store = store
        self._profiles = tuple(profiles)
        self._thresholds = thresholds
        self._skip = frozenset(skip)

    def find(self, document_id: str, message: str) -> tuple[ObjectFinding, Extraction] | None:
        """The finding and the detections it was counted from, or None."""
        question = classify(message)
        if question is None:
            return None
        for profile in self._profiles:
            if question.kind not in profile.classes or (profile.name, question.kind) in self._skip:
                continue
            try:
                extraction = self._store.get(document_id, profile.producer, profile.version)
            except Exception:  # a detections file that cannot be read must not lose the answer
                logging.getLogger(__name__).warning(
                    "The detections of %s cannot be read; the LLM answers", document_id, exc_info=True
                )
                return None
            if extraction is not None:
                return count(extraction, question, self._thresholds), extraction
        return None

    def solve(self, document_id: str, message: str) -> ObjectFinding | None:
        found = self.find(document_id, message)
        return found[0] if found is not None else None
