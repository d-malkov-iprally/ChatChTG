"""Recognise an object-count question about a Finnish plan and say what it counts."""

from __future__ import annotations

from dataclasses import dataclass

# Kind -> (the detector classes that it counts, the starts of its English and Finnish
# questions). The starts are those of the CubiCasa5K questions of the benchmark. The Swiss
# questions about the same objects are worded differently ("How many windows are drawn",
# "How many toilets (WC)", "How many showers are drawn") and are counted by other rules,
# so they are not claimed here.
_KINDS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "window": (("window",), ("how many windows are shown", "kuinka monta ikkunaa")),
    "door": (("door",), ("how many doors are shown", "kuinka monta ovea")),
    "toilet": (("toilet",), ("how many toilet bowls", "kuinka monta wc-istuinta")),
    "sink": (
        ("sink",),
        ("how many sinks and wash basins", "kuinka monta pesu- ja tiskiallasta"),
    ),
    "bath_shower": (
        ("shower", "bathtub"),
        ("how many showers and bathtubs", "kuinka monta suihkua ja kylpyammetta"),
    ),
}
KINDS = tuple(_KINDS)


@dataclass(frozen=True)
class ObjectQuestion:
    kind: str
    """One of KINDS."""

    @property
    def classes(self) -> tuple[str, ...]:
        """The detector classes whose objects the question counts together."""
        return _KINDS[self.kind][0]


def classify(question: str) -> ObjectQuestion | None:
    """Return what ``question`` counts, or None when it is not an object-count question."""
    lowered = " ".join(question.lower().split())
    for kind, (_, starts) in _KINDS.items():
        if lowered.startswith(starts):
            return ObjectQuestion(kind)
    return None
