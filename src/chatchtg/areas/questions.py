"""Read an area question and add up the rooms it asks for.

The benchmark asks four kinds of area question, in English, German, French and Italian:

- ``room``: the area of one room, given by its number.
- ``label``: the total of all rooms with a given name.
- ``apartment``: the net area of one apartment, which is its rooms without the outdoor ones.
- ``floor``: the total of all numbered indoor rooms of the floor.

The rules were checked against the reference answers:

- A sum uses the unrounded room areas.
- A floor total includes stairs, lifts and shafts, although the question says "excluding
  shafts". It excludes only the outdoor rooms.
- A label total includes outdoor rooms: "all rooms labelled 'Balkon'" is a valid question.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Literal

Kind = Literal["room", "label", "apartment", "floor"]


@dataclass
class AreaQuery:
    kind: Kind
    number: str | None = None
    label: str | None = None
    apartment: str | None = None


@dataclass
class RoomRow:
    """One row of the room table the pipeline builds."""

    number: str
    name: str | None
    area: float | None
    """Area in m², printed or measured. None when unknown."""
    apartment: str | None
    outdoor: bool


@dataclass
class AreaAnswer:
    value: float | None
    """None when a needed room is missing or has no area."""
    rooms: list[RoomRow] = field(default_factory=list)
    """The rooms that were added up."""
    missing: list[str] = field(default_factory=list)
    """Numbers of rooms that should count but have no area."""
    note: str = ""
    """One short plain sentence saying what was added up."""


_NUMBER = r"(?P<number>[A-Z]{0,2}\d+\.\d+)"
_OPEN = "['\"‘’‚“”„«‹]"
_CLOSE = "['\"‘’“”»›]"
_LETTER = r"(?P<apartment>[A-Z]{1,2})\b"


def _quoted(after: str) -> str:
    """A quoted label followed by ``after``. The label may contain an apostrophe."""
    return rf"{_OPEN}\s*(?P<label>.+?)\s*{_CLOSE}\s*{after}"


_TEMPLATES: list[tuple[Kind, str]] = [
    ("room", rf"\barea of room {_NUMBER}(?: \({_quoted(r'\)')})?"),
    ("room", rf"fläche von Raum {_NUMBER}(?: \({_quoted(r'\)')})?"),
    ("room", rf"\bsurface nette de la pièce {_NUMBER}(?: \({_quoted(r'\)')})?"),
    ("room", rf"\bsuperficie netta del locale {_NUMBER}(?: \({_quoted(r'\)')})?"),
    ("apartment", rf"\barea of apartment {_LETTER}"),
    ("apartment", rf"fläche von Wohnung {_LETTER}"),
    ("apartment", rf"\bsurface nette de l['’]appartement {_LETTER}"),
    ("apartment", rf"\bsuperficie netta dell['’]appartamento {_LETTER}"),
    ("floor", r"\ball numbered indoor rooms\b"),
    ("floor", r"\baller nummerierten Innenräume\b"),
    ("floor", r"\btoutes les pièces intérieures numérotées\b"),
    ("floor", r"\btutti i locali interni numerati\b"),
    ("label", rf"\ball rooms labell?ed {_quoted(r'(?:on this floor|\?|$)')}"),
    ("label", rf"\baller Räume mit der Bezeichnung {_quoted(r'(?:in diesem|\?|$)')}"),
    ("label", rf"\btoutes les pièces désignées {_quoted(r'(?:à cet|\?|$)')}"),
    ("label", rf"\btutti i locali denominati {_quoted(r'(?:in questo|\?|$)')}"),
]
_COMPILED = [(kind, re.compile(pattern)) for kind, pattern in _TEMPLATES]


def parse_question(text: str) -> AreaQuery | None:
    """Read an area question. Returns None when it is not one of the four templates."""
    text = " ".join(text.replace(" ", " ").replace(" ", " ").split())
    for kind, pattern in _COMPILED:
        match = pattern.search(text)
        if match:
            groups = match.groupdict()
            return AreaQuery(
                kind=kind,
                number=groups.get("number"),
                label=groups.get("label"),
                apartment=groups.get("apartment"),
            )
    return None


KNOWN_LABELS = (
    "Arkade", "Ascenseur", "Ascensore", "Bad", "Bagno", "Bain", "Balcon", "Balcone", "Balkon",
    "Buanderie", "Büro", "Bürofläche", "Cage d'escalier", "Camera da letto", "Chambre à coucher",
    "Compartiment cave", "Corridoio", "Couloir", "Cucina", "Cuisine", "Eingangshalle",
    "Einstellhalle", "Essen", "Fahrbahn", "Garten", "Giardino d'inverno", "Grossraumbüro", "Hall",
    "Hall d'entrée", "Halle", "Jardin d'hiver", "Kellerabteil", "Kinderwagen", "Korridor", "Küche",
    "Küche/Essen", "Lavanderia", "Lift", "Local", "Locale", "Locale tecnico", "Loggia", "Patio",
    "Pièce", "Pranzo", "Raum", "Reduit", "Repas", "Ripostiglio", "Réduit", "Schlafzimmer",
    "Sitzplatz", "Soggiorno", "Soggiorno/pranzo", "Stanza", "Surface de vente", "Séjour",
    "Séjour/repas", "Technik", "Terrasse", "Terrazza", "Treppenhaus", "Vano cantina", "Vano scala",
    "Velo/Kinderwagen", "Verkauf", "Waschen/Trocknen", "Wintergarten", "Wohnen", "Wohnen/Essen",
    "Zimmer",
)  # fmt: skip
"""Every room name of the Swiss Dwellings plans. A name that reads as one of these is never
taken for a misread of another: 'Balcon' is not 'Balkon' with a slip."""


def normalize_label(label: str) -> str:
    """Lower case, without accents, spaces or punctuation: 'Séjour / repas' -> 'sejourrepas'."""
    text = unicodedata.normalize("NFKD", label.replace("ß", "ss"))
    return "".join(ch for ch in text.casefold() if ch.isalnum())


_KNOWN = frozenset(normalize_label(label) for label in KNOWN_LABELS)


def _edit_distance(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb))
            )
        previous = current
    return previous[-1]


def _slips_allowed(length: int) -> int:
    if length < 3:
        return 0
    return 1 if length < 6 else 2


def label_matches(name: str | None, label: str) -> bool:
    """Whether a room name read from the plan is the label a question asks for.

    Accents, case, spaces and punctuation do not count. A name that is not a known room
    name may differ by one wrong character (two for a label of six letters or more), as
    long as no other known room name is nearer.
    """
    if not name:
        return False
    found, wanted = normalize_label(name), normalize_label(label)
    if found == wanted:
        return bool(wanted)
    if found in _KNOWN:
        return False
    distance = _edit_distance(found, wanted)
    if distance > _slips_allowed(len(wanted)):
        return False
    return all(_edit_distance(found, other) >= distance for other in _KNOWN if other != wanted)


def _normalize_number(number: str) -> str:
    return "".join(number.split()).upper().replace(",", ".")


def _select(query: AreaQuery, rooms: list[RoomRow]) -> tuple[list[RoomRow], str]:
    """The rooms a query counts, and the words that describe them."""
    if query.kind == "room":
        wanted = _normalize_number(query.number or "")
        found = [room for room in rooms if _normalize_number(room.number) == wanted]
        # A number read twice is one room: prefer the reading that has an area.
        found.sort(key=lambda room: room.area is None)
        return found[:1], f"room {query.number}"
    if query.kind == "label":
        label = query.label or ""
        return [room for room in rooms if label_matches(room.name, label)], f"rooms named '{label}'"
    if query.kind == "apartment":
        letter = (query.apartment or "").upper()
        found = [
            room
            for room in rooms
            if room.apartment is not None and room.apartment.upper() == letter and not room.outdoor
        ]
        return found, f"indoor rooms of apartment {query.apartment}"
    return [room for room in rooms if not room.outdoor], "numbered indoor rooms of the floor"


def answer_query(query: AreaQuery, rooms: list[RoomRow]) -> AreaAnswer:
    """Add up the rooms a query asks for.

    The value is None when no room fits the query or when one of the rooms has no area.
    """
    counted, what = _select(query, rooms)
    if not counted:
        missing = [query.number] if query.kind == "room" and query.number else []
        return AreaAnswer(value=None, missing=missing, note=f"Found no {what} in the room table.")
    missing = [room.number for room in counted if room.area is None]
    if missing:
        return AreaAnswer(
            value=None,
            rooms=counted,
            missing=missing,
            note=f"The {what} cannot be added up: no area for {', '.join(missing)}.",
        )
    value = float(sum(room.area for room in counted if room.area is not None))
    if query.kind == "room":
        note = f"Area of {what}."
    else:
        numbers = ", ".join(room.number for room in counted)
        note = f"Sum of the {len(counted)} {what}: {numbers}."
    return AreaAnswer(value=value, rooms=counted, note=note)
