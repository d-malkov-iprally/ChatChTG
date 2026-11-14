"""The detector interface and the helpers that do not need a model."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Protocol

from PIL import Image


@dataclass(frozen=True)
class Detection:
    label: str
    score: float
    """The model's confidence, from 0 to 1. Scores of different models are not comparable."""
    box: tuple[float, float, float, float]
    """(left, top, right, bottom) in pixels of the image."""


class Detector(Protocol):
    def detect(self, image: Image.Image, labels: list[str], threshold: float) -> list[Detection]:
        """Return the objects named by ``labels`` with a score of at least ``threshold``."""
        ...


def parse_labels(prompt: str) -> list[str]:
    """Split a prompt such as "window, door" into labels. Duplicates and case are removed."""
    labels: list[str] = []
    for part in prompt.replace("\n", ",").replace(".", ",").split(","):
        label = " ".join(part.lower().split())
        if label and label not in labels:
            labels.append(label)
    return labels


def count_by_label(detections: list[Detection]) -> dict[str, int]:
    """Count the detections of each label, most frequent first."""
    return dict(Counter(detection.label for detection in detections).most_common())
