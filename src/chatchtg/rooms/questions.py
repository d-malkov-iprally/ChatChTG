"""Recognise a room-count question and say what it counts."""

from __future__ import annotations

import re
from dataclasses import dataclass

# The kinds of room-count question. The first seven are asked about Finnish plans
# (CubiCasa5K), the last two about Swiss plans (Swiss Dwellings).
FINNISH_KINDS = ("bedroom", "sauna", "kitchen", "bathroom", "wc", "balcony", "storeys")
SWISS_KINDS = ("label", "apartments")

_FINNISH = {
    "bedroom": ("how many bedrooms", "kuinka monta makuuhuonetta"),
    "sauna": ("how many saunas", "kuinka monta saunaa"),
    "kitchen": ("how many kitchens", "kuinka monta keittiötä"),
    "bathroom": ("how many bathrooms or washrooms", "kuinka monta kylpy- tai pesuhuonetta"),
    "wc": ("how many separate wc rooms", "kuinka monta erillistä wc-tilaa"),
    "balcony": ("how many balconies", "kuinka monta parveketta"),
    "storeys": ("how many floors (storeys)", "kuinka monta kerrosta"),
}
_APARTMENTS = (
    "how many apartments",
    "wie viele wohnungen",
    "combien d'appartements",
    "quanti appartamenti",
)
_LABEL = (
    re.compile(r"how many rooms labelled '(.+)' are there", re.I),
    re.compile(r"wie viele räume mit der bezeichnung «\s*(.+?)\s*»", re.I),
    re.compile(r"combien de pièces portant la désignation «\s*(.+?)\s*»", re.I),
    re.compile(r"quanti locali con la denominazione «\s*(.+?)\s*»", re.I),
)


@dataclass(frozen=True)
class RoomQuestion:
    kind: str
    """One of FINNISH_KINDS or SWISS_KINDS."""
    label: str | None = None
    """The room label that a ``label`` question asks about."""

    @property
    def profile(self) -> str:
        """Which kind of plan the question is about: ``finnish`` or ``swiss``."""
        return "swiss" if self.kind in SWISS_KINDS else "finnish"


def classify(question: str) -> RoomQuestion | None:
    """Return what ``question`` counts, or None when it is not a room-count question."""
    text = question.strip()
    lowered = text.lower().replace("’", "'")
    for kind, prefixes in _FINNISH.items():
        if lowered.startswith(prefixes):
            return RoomQuestion(kind)
    if lowered.startswith(_APARTMENTS):
        return RoomQuestion("apartments")
    for pattern in _LABEL:
        match = pattern.search(text)
        if match:
            return RoomQuestion("label", match.group(1))
    return None
