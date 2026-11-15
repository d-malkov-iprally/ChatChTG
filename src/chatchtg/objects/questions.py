"""Recognise an object-count question and say what kind of object it counts."""

from __future__ import annotations

from dataclasses import dataclass

# Kind -> the starts of its questions, lowercased. They are the starts of the object-count
# templates of the benchmark: the CubiCasa5K (Finnish plans) templates in English and Finnish,
# the Swiss Dwellings templates in English, German, French and Italian. Which detector classes
# a kind counts depends on the detector (``counter.DetectorProfile.classes``).
#
# The Finnish and the Swiss templates about windows, doors and toilets ask for the same
# objects, so they are one kind each. "Sinks and wash basins (kitchen and bathroom combined)"
# and "showers and bathtubs together" are Finnish templates; "washbasins", "bathtubs" and
# "showers" alone are Swiss templates.
_STARTS: dict[str, tuple[str, ...]] = {
    "window": (
        "how many windows are shown",
        "kuinka monta ikkunaa",
        "how many windows are drawn",
        "wie viele fenster sind",
        "combien de fenêtres sont",
        "quante finestre sono",
    ),
    "door": (
        "how many doors are shown",
        "kuinka monta ovea",
        "how many doors (interior doors and entrance doors together)",
        "wie viele türen (innentüren und eingangstüren zusammen)",
        "combien de portes (portes intérieures et portes d",
        "quante porte (porte interne e porte d",
    ),
    "toilet": (
        "how many toilet bowls",
        "kuinka monta wc-istuinta",
        "how many toilets (wc) are",
        "wie viele wcs sind",
        "combien de wc sont",
        "quanti wc sono",
    ),
    "sink": ("how many sinks and wash basins", "kuinka monta pesu- ja tiskiallasta"),
    "washbasin": (
        "how many washbasins are",
        "wie viele lavabos sind",
        "combien de lavabos sont",
        "quanti lavabi sono",
    ),
    "bath_shower": ("how many showers and bathtubs", "kuinka monta suihkua ja kylpyammetta"),
    "bathtub": (
        "how many bathtubs are",
        "wie viele badewannen sind",
        "combien de baignoires sont",
        "quante vasche da bagno sono",
    ),
    "shower": (
        "how many showers are",
        "wie viele duschen sind",
        "combien de douches sont",
        "quante docce sono",
    ),
}
KINDS = tuple(_STARTS)


@dataclass(frozen=True)
class ObjectQuestion:
    kind: str
    """One of KINDS."""


def classify(question: str) -> ObjectQuestion | None:
    """Return what ``question`` counts, or None when it is not an object-count question."""
    lowered = " ".join(question.lower().split())
    for kind, starts in _STARTS.items():
        if lowered.startswith(starts):
            return ObjectQuestion(kind)
    return None
