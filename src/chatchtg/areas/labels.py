"""Read the room labels of a rendered floor plan with local OCR.

A room label is a centred stack of up to three lines: a name ("Küche", may be missing),
a bold room number ("7.16", "U1.03") and, on some plans, the printed area ("15.61 m²").
Apartments carry a boxed tag ("Whg. A", "App. B"). The title block states the scale.

How a plan is read:

1. The sheet frame is found; the title block is read once for the scale. The drawing is
   cut into overlapping tiles, each enlarged and passed to RapidOCR (text detection and
   recognition).
2. The text lines are grouped into labels around the room numbers.
3. The renderer numbers the rooms of a plan ``<floor>.01 .. <floor>.NN`` without gaps, and
   consecutive numbers are neighbours. That drops stray numbers, gives damaged number
   lines the missing number they fit, and names the numbers that are still missing.
4. Where a label is incomplete (a line without a number, a number without name or area)
   and around the neighbours of every missing number, the plan is read again: enlarged
   more, with long straight lines removed, and once more with thin strokes faded out so
   that only the bold numbers remain.

Nothing here uses the DXF twin of a plan: the room-name vocabulary is static knowledge
about the renderer.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import Any

import cv2
import numpy as np

Box = tuple[float, float, float, float]

ROOM_NAMES: dict[str, tuple[str, ...]] = {
    "de": (
        "Zimmer", "Bad", "Korridor", "Küche", "Balkon", "Wohnen/Essen", "Schlafzimmer", "Reduit",
        "Treppenhaus", "Wohnen", "Lift", "Loggia", "Raum", "Essen", "Terrasse", "Büro",
        "Waschen/Trocknen", "Halle", "Sitzplatz", "Wintergarten", "Bürofläche", "Garten",
        "Küche/Essen", "Einstellhalle", "Verkauf", "Kellerabteil", "Arkade", "Kinderwagen",
        "Eingangshalle", "Grossraumbüro", "Velo/Kinderwagen", "Technik", "Fahrbahn",
    ),
    "fr": (
        "Pièce", "Bain", "Couloir", "Cuisine", "Balcon", "Séjour/repas", "Chambre à coucher",
        "Réduit", "Cage d'escalier", "Séjour", "Loggia", "Ascenseur", "Repas", "Local",
        "Compartiment cave", "Jardin d'hiver", "Terrasse", "Patio", "Buanderie", "Hall d'entrée",
        "Hall", "Surface de vente",
    ),
    "it": (
        "Stanza", "Bagno", "Corridoio", "Cucina", "Balcone", "Soggiorno/pranzo",
        "Camera da letto", "Ripostiglio", "Vano scala", "Soggiorno", "Ascensore", "Loggia",
        "Vano cantina", "Pranzo", "Terrazza", "Lavanderia", "Locale tecnico", "Locale",
        "Giardino d'inverno",
    ),
}
"""Every room name the renderer prints, per plan language, most common first."""

TILE = 700
"""Side of an OCR tile in plan pixels."""
OVERLAP = 140
"""Tiles share this many pixels, so that a label cut by one tile is whole in the next."""
ZOOM = 2.0
"""Tiles are enlarged by this factor: label text is only 6 to 12 px high."""
REREAD_ZOOM = 4.0
"""Enlargement of the patch around an incomplete label when it is read again."""
MAX_REREADS = 40
"""At most this many incomplete labels are read again per round."""
GAP_ZOOM = 3.0
"""Enlargement of the windows searched for a missing room number."""
MAX_GAP_WINDOWS = 12
"""At most this many such windows per round."""
CANVAS = 1400
"""Side in pixels of the largest image given to the OCR engine."""
LINE_RUN = 15
"""A straight horizontal or vertical run of ink this long is a drawn line, not text."""
BOLD_STROKE = 1.7
"""Strokes thinner than this many plan pixels are faded out to leave only bold text."""
WEAK_FIT_REACH = 0.5
"""A line with too few digits left to tell its number still gets a missing number when it
is this close to the label of the number before or after, in typical label spacings."""
RED_MINUS_BLUE = 60
"""A pixel of the red apartment outline has at least this much more red than blue."""

OCR_THREADS: int | None = None
"""Threads per OCR model; None lets onnxruntime take every core. Set it before the first
read when several processes read plans side by side."""

# Sheet layout of the renderer, used when the frame lines are not found: the frame sits
# 58 px from the sheet edge and the legend and title block fill the bottom 250 px of it.
FRAME_MARGIN = 58
BOTTOM_BLOCK = 250

_NUMBER = re.compile(r"^([A-Z]?\d{1,2})\s?[.,·:]\s?(\d{2,3})$")
_AREA = re.compile(r"^(\d{1,4})\s?[.,]\s?(\d{2})\s*(m.{0,2})?$")
_AREA_LENIENT = re.compile(r"^(\d{1,4})[.,:]?(\d{2})[\s/|]*[mM][²2a-zA-Z?]?\.?$")
_TAG = re.compile(r"^[\[(|]?(?:[Ww]h[a-z]?|[Ww][bh]g|[Aa]p[a-z]?)\s*[.,:;]?\s*([A-Z])[\])|]?$")
_SCALE = re.compile(r"1\s?[:;]\s?(100|200)(?!\d)")
_LOOKALIKES = str.maketrans(
    {"O": "0", "Q": "0", "o": "0", "D": "0", "l": "1", "I": "1", "|": "1", "i": "1", "y": "1",
     "S": "5", "s": "5", "B": "8", "Z": "2", "z": "2", "g": "9", "b": "6"}
)


@dataclass
class RoomLabel:
    number: str
    """Normalised room number, e.g. "7.16"."""
    name: str | None
    """As read; snapped to ``ROOM_NAMES`` when close to one of them."""
    printed_area: float | None
    """In m², when the label has a third line."""
    seed: tuple[float, float]
    """A point inside the room: between the name line and the number line, or just above
    the number when there is no name."""
    number_xy: tuple[float, float]
    """Centre of the number text."""
    number_certain: bool = True
    """False when the number was not read cleanly but fitted to the numbering sequence."""


@dataclass
class PlanLabels:
    scale: int | None
    """100 or 200."""
    rooms: list[RoomLabel] = field(default_factory=list)
    tags: dict[str, tuple[float, float]] = field(default_factory=dict)
    """Apartment letter -> centre of its tag."""
    text_boxes: list[Box] = field(default_factory=list)
    """(x0, y0, x1, y1) of every label text line and tag inside the drawing."""
    missing_numbers: list[str] = field(default_factory=list)
    """Gaps in the numbering sequence: rooms that were probably missed."""


@dataclass(frozen=True)
class TextLine:
    """One line of text found by OCR, with its box in plan pixels."""

    text: str
    box: Box
    confidence: float = 1.0

    @property
    def cx(self) -> float:
        return (self.box[0] + self.box[2]) / 2

    @property
    def cy(self) -> float:
        return (self.box[1] + self.box[3]) / 2

    @property
    def width(self) -> float:
        return self.box[2] - self.box[0]

    @property
    def height(self) -> float:
        return self.box[3] - self.box[1]


# ----------------------------------------------------------------------------------------
# Text normalisation


def fold(text: str) -> str:
    """Lower case without accents, for comparing names."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c)).strip()


def _squash(text: str) -> str:
    """Letters only; "ü" is often read as "ii"."""
    return re.sub(r"[^a-z]", "", fold(text)).replace("ii", "u")


_SQUASHED_NAMES: dict[str | None, dict[str, str]] = {
    language: {_squash(name): name for name in reversed(names)}
    for language, names in ROOM_NAMES.items()
}
_SQUASHED_NAMES[None] = {
    _squash(name): name for names in reversed(ROOM_NAMES.values()) for name in reversed(names)
}
_ALL_NAMES = frozenset(name for names in ROOM_NAMES.values() for name in names)
_OWN_LANGUAGE = {
    squashed: language
    for language in ROOM_NAMES
    for squashed in _SQUASHED_NAMES[language]
    if sum(squashed in _SQUASHED_NAMES[other] for other in ROOM_NAMES) == 1
}
NAME_CUTOFF = 0.74
NAME_CUTOFF_IN_LANGUAGE = 0.5
"""How similar a read name must be to a vocabulary name to be replaced by it. Once the
language of the plan is known the choice is small and a loose match is safe."""


COMMON_NAMES = 8
COMMON_NAME_BONUS = 0.2


def plan_language(texts: list[str]) -> str | None:
    """The language of a plan's room names: "de", "fr" or "it"."""
    votes = Counter(_OWN_LANGUAGE[s] for text in texts if (s := _squash(text)) in _OWN_LANGUAGE)
    return votes.most_common(1)[0][0] if votes else None


def snap_name(text: str, language: str | None = None, cutoff: float | None = None) -> str | None:
    """The vocabulary name closest to ``text``, or None when none is close."""
    if cutoff is None:
        cutoff = NAME_CUTOFF if language is None else NAME_CUTOFF_IN_LANGUAGE
    exact = text.strip()
    if exact in (ROOM_NAMES[language] if language else _ALL_NAMES):
        return exact
    names = _SQUASHED_NAMES[language]
    squashed = _squash(text)
    if len(squashed) < 2:
        return None
    if squashed in names:
        return names[squashed]
    best, best_score = None, 0.0
    for candidate, name in names.items():
        ratio = difflib.SequenceMatcher(None, squashed, candidate).ratio()
        # A badly read name is far more often a bathroom than an office.
        common = language is not None and name in ROOM_NAMES[language][:COMMON_NAMES]
        score = ratio + (COMMON_NAME_BONUS if common else 0.0)
        if ratio >= cutoff and score >= best_score:
            best, best_score = name, score
    return best


def parse_number(text: str) -> str | None:
    """A cleanly read room number: "7.16", "U1.03", "12.07", "1.104"."""
    match = _NUMBER.match(text.strip().replace(" ", ""))
    return f"{match.group(1)}.{match.group(2)}" if match else None


def parse_area(text: str) -> tuple[float, bool] | None:
    """The area printed in a label line, and whether its unit was read.

    Without a unit ("15.61") the line may just as well be a room number.
    """
    stripped = text.strip()
    match = _AREA.match(stripped)
    if match:
        return float(f"{match.group(1)}.{match.group(2)}"), bool(match.group(3))
    # "/4.50m²", "2566/m2", "868 m?": anything that ends in the unit.
    match = _AREA_LENIENT.match(re.sub(r"^[^0-9]+", "", stripped))
    if match:
        return float(f"{match.group(1)}.{match.group(2)}"), True
    return None


def parse_tag(text: str) -> str | None:
    """The apartment letter of a tag such as "Whg. A" or "App. B"."""
    match = _TAG.match(text.strip())
    return match.group(1) if match else None


def parse_scale(text: str) -> int | None:
    match = _SCALE.search(text)
    return int(match.group(1)) if match else None


def repair_number(text: str, prefix: str) -> str | None:
    """Read a damaged room number, knowing the floor prefix all numbers of the plan share.

    Handles a wrong or missing separator ("3:38", "338", "4.10."), look-alike characters
    ("4.Q2", "U1.O3") and a letter prefix read as something else.
    """
    stripped = text.strip().replace(" ", "")
    if not stripped or len(stripped) > len(prefix) + 6:
        return None
    if re.search(r"m[²2a?]?$", stripped) and len(stripped) > len(prefix) + 3:
        return None
    letter = prefix[0] if prefix[0].isalpha() else ""
    if letter:
        if stripped[0].upper() not in {letter, "V", "0", "O"} or len(stripped) < 2:
            return None
        stripped = stripped[1:]
    if sum(c.isalpha() for c in stripped) > 1:
        return None
    digits = re.sub(r"[^0-9]", "", stripped.translate(_LOOKALIKES))
    floor = prefix[len(letter):]
    if not digits.startswith(floor):
        return None
    suffix = digits[len(floor):]
    if len(suffix) not in (2, 3) or (len(suffix) == 3 and suffix[0] == "0"):
        return None
    # A separator the reader did see must sit where the prefix ends.
    seen = re.search(r"[.,·:]", stripped.rstrip(".,·:"))
    if seen and len(re.sub(r"[^0-9]", "", stripped[: seen.start()].translate(_LOOKALIKES))) != len(floor):
        return None
    return f"{prefix}.{suffix}"


def _split_number(number: str) -> tuple[str, int]:
    prefix, suffix = number.split(".")
    return prefix, int(suffix)


def _format_number(prefix: str, index: int) -> str:
    return f"{prefix}.{index:02d}"


# ----------------------------------------------------------------------------------------
# Table building


def _stacked(upper: TextLine, lower: TextLine, low: float = 0.4, high: float = 2.6) -> bool:
    """Whether ``lower`` is the next line of the label that ``upper`` belongs to."""
    height = max(upper.height, lower.height)
    dy = lower.cy - upper.cy
    reach = 0.5 * max(upper.width, lower.width) + 0.3 * height
    return low * height < dy < high * height and abs(lower.cx - upper.cx) < reach


def merge_fragments(lines: list[TextLine]) -> list[TextLine]:
    """Join short pieces of one text line that the reader split, e.g. "4." + "24"."""
    ordered = sorted(lines, key=lambda line: line.box[0])
    used: set[int] = set()
    merged: list[TextLine] = []
    for i, left in enumerate(ordered):
        if i in used:
            continue
        for j, right in enumerate(ordered):
            if j == i or j in used or len(left.text) > 3 or len(right.text) > 3:
                continue
            height = max(left.height, right.height)
            gap = right.box[0] - left.box[2]
            if right.cx > left.cx and abs(right.cy - left.cy) < 0.35 * height and gap < 0.6 * height:
                box = (left.box[0], min(left.box[1], right.box[1]), right.box[2], max(left.box[3], right.box[3]))
                left = TextLine(left.text + right.text, box, min(left.confidence, right.confidence))
                used.add(j)
                break
        merged.append(left)
    return merged


def plan_prefix(lines: list[TextLine]) -> str | None:
    """The floor prefix shared by the room numbers of a plan."""
    counts: Counter[str] = Counter()
    for line in lines:
        number = parse_number(line.text)
        area = parse_area(line.text)
        if number and not (area and area[1]):
            counts[number.split(".")[0]] += 1
    if not counts:
        return None
    best, count = counts.most_common(1)[0]
    return best if count >= 2 or len(counts) == 1 else None


def _edit_distance(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[-1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def _digits(text: str) -> str:
    return re.sub(r"[^0-9]", "", text.translate(_LOOKALIKES))


def _distance(a: TextLine, b: TextLine) -> float:
    return max(abs(a.cx - b.cx), abs(a.cy - b.cy))


@dataclass
class _Row:
    number: str
    line: TextLine
    name: TextLine | None = None
    area: TextLine | None = None
    certain: bool = True
    """False when the number was repaired or inferred rather than read."""


@dataclass
class Table:
    """The labels of a plan, still tied to the text lines they were read from."""

    prefix: str | None
    rows: list[_Row] = field(default_factory=list)
    tags: list[tuple[str, TextLine]] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    orphans: list[TextLine] = field(default_factory=list)
    """Label-like lines that belong to no row: probably a room whose number was missed."""
    lines: list[TextLine] = field(default_factory=list)
    language: str | None = None

    @property
    def prints_areas(self) -> bool:
        return sum(row.area is not None for row in self.rows) >= max(2, 0.3 * len(self.rows))

    def neighbour_spacing(self) -> float:
        """Typical distance in pixels between the labels of consecutive room numbers."""
        by_index = {_split_number(row.number)[1]: row.line for row in self.rows}
        steps = [_distance(line, by_index[k + 1]) for k, line in by_index.items() if k + 1 in by_index]
        return float(np.median(steps)) if steps else 150.0


def _is_name(line: TextLine) -> bool:
    text = line.text
    if sum(c.isalpha() for c in text) < 2 or parse_tag(text) or re.search(r"1\s?:\s?\d", text):
        return False
    return parse_area(text) is None


def _drop_runaways(numbers: dict[str, int]) -> dict[str, int]:
    """Drop numbers far beyond the end of the sequence, e.g. a "2.728" among 60 rooms.

    Rooms are numbered 1..N and most are read, so N cannot be much above the count.
    """
    kept = dict(numbers)
    while kept:
        last = max(kept, key=lambda n: _split_number(n)[1])
        if _split_number(last)[1] <= 1.6 * (len(kept) - 1) + 3:
            break
        del kept[last]
    return kept


def _is_subsequence(part: str, whole: str) -> bool:
    rest = iter(whole)
    return all(c in rest for c in part)


def _fill_gaps(
    lines: list[TextLine], winners: dict[str, int], leftovers: list[int], prefix: str
) -> dict[str, int]:
    """Give damaged number lines the missing numbers of the sequence they fit.

    A line fits a number when at most one digit differs ("7.64" for a missing 7.04 when
    7.64 is taken, "5.6B" for 5.63, "16" for 3.16) and it lies near the labels of the
    numbers just before or after it, as consecutive numbers are neighbours on the plan.
    Being the only fit for each other makes up for being far, and being right next to
    the neighbouring numbers makes up for most digits being lost ("3." for 3.20).
    """
    by_index = {_split_number(number)[1]: lines[i] for number, i in winners.items()}
    if not by_index or not leftovers:
        return {}
    steps = [_distance(line, by_index[k + 1]) for k, line in by_index.items() if k + 1 in by_index]
    spacing = float(np.median(steps)) if steps else 150.0
    last = max(by_index)
    floor = prefix.lstrip("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    gaps = [k for k in range(1, last + 1) if k not in by_index] + [last + 1]
    reads = {i: _digits(lines[i].text) for i in leftovers}

    def differ(i: int, k: int) -> float:
        wanted = floor + f"{k:02d}"
        if floor + reads[i] == wanted:
            return 0.5  # the prefix was lost: ".16", "16"
        return float(_edit_distance(reads[i], wanted)) if len(reads[i]) >= 2 else 9.0

    def away(i: int, k: int, reach: int) -> float:
        near = [by_index[j] for j in range(k - reach, k + reach + 1) if j in by_index]
        return min((_distance(lines[i], other) for other in near), default=1.5 * spacing) / spacing

    filled: dict[str, int] = {}
    taken: set[int] = set()

    def assign(options: list[tuple[float, int, int]]) -> None:
        for _, k, i in sorted(options):
            number = _format_number(prefix, k)
            if number not in filled and i not in taken:
                filled[number] = i
                taken.add(i)

    fits = [(i, k) for k in gaps for i in leftovers if differ(i, k) <= 1]
    assign([(cost, k, i) for i, k in fits if (cost := differ(i, k) + 0.5 * min(away(i, k, 2), 3.0)) <= 2.0])
    open_fits = [(i, k) for i, k in fits if i not in taken and _format_number(prefix, k) not in filled]
    per_line, per_gap = Counter(i for i, _ in open_fits), Counter(k for _, k in open_fits)
    assign([(0.0, k, i) for i, k in open_fits if per_line[i] == 1 and per_gap[k] == 1 and k <= last])
    weak = [
        (distance, k, i)
        for k in gaps[:-1]
        for i in leftovers
        if i not in taken and reads[i] and lines[i].confidence >= 0.5
        and _is_subsequence(reads[i], floor + f"{k:02d}")
        and (distance := away(i, k, 1)) <= WEAK_FIT_REACH
    ]
    assign(weak)
    return filled


def build_table(lines: list[TextLine]) -> Table:
    """Group text lines into room labels and check them against the numbering sequence."""
    lines = merge_fragments(lines)
    prefix = plan_prefix(lines)
    language = plan_language([line.text for line in lines])
    table = Table(prefix=prefix, lines=lines, language=language)
    table.tags = [(letter, line) for line in lines if (letter := parse_tag(line.text))]
    if prefix is None:
        return table
    tag_lines = {i for i, line in enumerate(lines) if parse_tag(line.text)}

    areas = {i: area for i, line in enumerate(lines) if (area := parse_area(line.text))}
    numbers: dict[int, tuple[str, bool]] = {}
    for i, line in enumerate(lines):
        if i in areas and areas[i][1]:
            continue
        clean = parse_number(line.text)
        if clean and clean.split(".")[0] == prefix:
            numbers[i] = (clean, True)
        elif i not in areas and (repaired := repair_number(line.text, prefix)):
            numbers[i] = (repaired, False)
    # A unit-less "4.96" directly below a room number is that room's area, also on floor 4;
    # unless it continues the numbering: small rooms without a name can be stacked closely.
    below_number: dict[int, str] = {}
    for i in sorted(numbers, key=lambda k: lines[k].cy):
        if i not in numbers:
            continue
        for j in list(areas):
            if j != i and not areas[j][1] and _stacked(lines[i], lines[j]):
                if j in numbers:
                    below_number[j] = numbers.pop(j)[0]
                else:
                    below_number[j] = ""

    # One row per number: the cleanly read and more confident line wins.
    winners: dict[str, int] = {}
    for i, (number, certain) in numbers.items():
        if _split_number(number)[1] == 0:
            continue  # numbering starts at .01
        j = winners.get(number)
        if j is None or (certain, lines[i].confidence) > (numbers[j][1], lines[j].confidence):
            winners[number] = i
    winners = _drop_runaways(winners)
    if winners:
        last = max(_split_number(number)[1] for number in winners)
        for j, number in sorted(below_number.items(), key=lambda item: item[1]):
            if number and number not in winners and _split_number(number)[1] <= last + 1:
                winners[number] = j
                numbers[j] = (number, True)
                del below_number[j]
                last = max(last, _split_number(number)[1])
    certain = {number: numbers[i][1] for number, i in winners.items()}

    taken = set(winners.values())
    leftovers = [
        i for i, line in enumerate(lines)
        if i not in taken and i not in tag_lines and i not in below_number
        and not (i in areas and areas[i][1])
        and (i in numbers or re.search(r"\d", line.text))
        and len(line.text.strip()) <= 8 and sum(c.isalpha() for c in line.text) <= 2
    ]
    for number, i in _fill_gaps(lines, winners, leftovers, prefix).items():
        winners[number] = i
        certain[number] = False

    used: set[int] = set(winners.values())
    rows: list[_Row] = []
    for number, i in sorted(winners.items(), key=lambda item: (lines[item[1]].cy, lines[item[1]].cx)):
        row = _Row(number=number, line=lines[i], certain=certain[number])
        below = [j for j in areas if j not in used and _stacked(lines[i], lines[j])]
        if below:
            j = min(below, key=lambda k: lines[k].cy)
            row.area = lines[j]
            used.add(j)
        rows.append(row)
    for row in rows:
        above = [
            j for j, line in enumerate(lines)
            if j not in used and j not in tag_lines and _is_name(line) and _stacked(line, row.line)
        ]
        if above:
            j = max(above, key=lambda k: (snap_name(lines[k].text, language) is not None, lines[k].cy))
            if snap_name(lines[j].text, language) or sum(c.isalpha() for c in lines[j].text) >= 4:
                row.name = lines[j]
                used.add(j)
    table.rows = rows

    present = {_split_number(row.number)[1] for row in rows}
    if present:
        table.missing = [
            _format_number(prefix, k) for k in range(1, max(present) + 1) if k not in present
        ]
    for j, line in enumerate(lines):
        if j in used or j in tag_lines:
            continue
        area = areas.get(j)
        if (area and area[1]) or j in leftovers or (_is_name(line) and snap_name(line.text, language, 0.8)):
            table.orphans.append(line)
    return table


def _seed(row: _Row) -> tuple[float, float]:
    number = row.line
    if row.name is not None:
        return number.cx, (row.name.box[3] + number.box[1]) / 2
    return number.cx, number.box[1] - 0.25 * number.height


def to_plan_labels(table: Table, scale: int | None) -> PlanLabels:
    rooms = []
    boxes: list[Box] = []
    for row in table.rows:
        name = None
        if row.name is not None:
            name = snap_name(row.name.text, table.language) or row.name.text.strip()
        area = parse_area(row.area.text) if row.area is not None else None
        rooms.append(
            RoomLabel(
                number=row.number,
                name=name,
                printed_area=area[0] if area else None,
                seed=_seed(row),
                number_xy=(row.line.cx, row.line.cy),
                number_certain=row.certain,
            )
        )
        boxes += [line.box for line in (row.name, row.line, row.area) if line is not None]
    tags: dict[str, tuple[float, float]] = {}
    for letter, line in sorted(table.tags, key=lambda tag: tag[1].confidence):
        tags[letter] = (line.cx, line.cy)
        boxes.append(line.box)
    # Orphans are erased as text too, unless they may be a fixture or a wall corner
    # mistaken for a digit: erasing ink there would open a room.
    for line in table.orphans:
        if not re.fullmatch(r"\W*\w?\W*|[oO0\W]+", line.text) and line.confidence >= 0.6:
            boxes.append(line.box)
    return PlanLabels(
        scale=scale, rooms=rooms, tags=tags, text_boxes=boxes, missing_numbers=list(table.missing)
    )


# ----------------------------------------------------------------------------------------
# OCR


@lru_cache(maxsize=1)
def _engine() -> Any:
    from rapidocr_onnxruntime import RapidOCR

    if OCR_THREADS is None:
        return RapidOCR()
    return RapidOCR(intra_op_num_threads=OCR_THREADS, inter_op_num_threads=1)


def _ocr(image: np.ndarray, zoom: float, origin: tuple[float, float]) -> list[TextLine]:
    """Text lines of an image patch, in plan pixels. ``origin`` is the patch's corner."""
    if zoom != 1:
        image = cv2.resize(image, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC)
    result, _ = _engine()(image, use_cls=False)
    lines = []
    for corners, text, confidence in result or []:
        points = np.asarray(corners, float) / zoom
        x0, y0 = points.min(axis=0) + origin
        x1, y1 = points.max(axis=0) + origin
        lines.append(TextLine(text, (float(x0), float(y0), float(x1), float(y1)), float(confidence)))
    return lines


@dataclass(frozen=True)
class Layout:
    """Where the parts of the sheet are, as (x0, y0, x1, y1) pixel boxes."""

    drawing: tuple[int, int, int, int]
    bottom: tuple[int, int, int, int]
    """The legend and the title block."""


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """(first, last) index of every run of True."""
    padded = np.concatenate([[False], flags, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(a), int(b) - 1) for a, b in zip(edges[::2], edges[1::2])]


def find_layout(image: np.ndarray) -> Layout:
    """Find the sheet frame and the line that separates the drawing from the legend.

    Sheets are A3, landscape or portrait. The frame and the separator are the only lines
    that run across the whole sheet.
    """
    height, width = image.shape[:2]
    dark = image.min(axis=2) < 128
    rows = _runs(dark.mean(axis=1) > 0.7)
    columns = _runs(dark.mean(axis=0) > 0.7)
    if len(rows) >= 3 and len(columns) >= 2:
        top, separator, bottom = rows[0][1] + 1, rows[-2], rows[-1][0]
        left, right = columns[0][1] + 1, columns[-1][0]
        return Layout((left, top, right, separator[0]), (left, separator[1] + 1, right, bottom))
    m = FRAME_MARGIN
    split = height - m - BOTTOM_BLOCK
    return Layout((m, m, width - m, split), (m, split, width - m, height - m))


def _without_red(image: np.ndarray) -> np.ndarray:
    """The image with the red apartment outline, which crosses labels, painted white."""
    red = image[..., 0].astype(np.int16) - image[..., 2] > RED_MINUS_BLUE
    cleaned = image.copy()
    cleaned[red] = 255
    return cleaned


def _without_long_lines(image: np.ndarray) -> np.ndarray:
    """The image with every long horizontal or vertical line painted white.

    Walls, fixtures and stair treads run through many labels. No stroke of the label text
    is as long as ``LINE_RUN`` pixels, so a straight run that long is not text. Where a
    line touches other ink it is kept, so that a glyph it crosses stays in one piece.
    """
    dark = (image.min(axis=2) < 160).astype(np.uint8)
    horizontal = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((1, LINE_RUN), np.uint8))
    vertical = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((LINE_RUN, 1), np.uint8))
    lines = horizontal | vertical
    other = dark & ~lines
    erase = lines & ~cv2.dilate(other, np.ones((3, 3), np.uint8))
    halo = cv2.dilate(erase, np.ones((3, 3), np.uint8)) & ~dark
    cleaned = image.copy()
    cleaned[(erase | halo).astype(bool)] = 255
    return cleaned


def _bold_only(patch: np.ndarray, zoom: float) -> np.ndarray:
    """Fade out thin strokes of an enlarged patch, to leave the bold room numbers.

    Room numbers are printed bold; the lines that cross them (the cross of a lift shaft, a
    door swing, a fixture) are one pixel thin. Names and areas fade as well.
    """
    size = max(3, int(round(BOLD_STROKE * zoom)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    closed = cv2.morphologyEx(patch, cv2.MORPH_CLOSE, kernel).astype(np.float32)
    return np.clip((closed - 80.0) * (255.0 / 90.0), 0, 255).astype(np.uint8)


def _tile_starts(low: int, high: int) -> list[int]:
    span = high - low
    if span <= TILE:
        return [low]
    count = int(np.ceil((span - OVERLAP) / (TILE - OVERLAP)))
    step = (span - TILE) / (count - 1)
    return [int(round(low + k * step)) for k in range(count)]


def _overlap_share(a: Box, b: Box) -> float:
    """Intersection over the smaller of the two boxes."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return w * h / max(smaller, 1e-9)


def _dedupe(lines: list[TextLine]) -> list[TextLine]:
    kept: list[TextLine] = []
    for line in sorted(lines, key=lambda l: -l.confidence):
        if all(_overlap_share(line.box, other.box) < 0.5 for other in kept):
            kept.append(line)
    return kept


def detect_lines(image: np.ndarray, area: tuple[int, int, int, int]) -> list[TextLine]:
    """First pass: every text line inside the drawing area."""
    x0, y0, x1, y1 = area
    ink = image[y0:y1, x0:x1].min(axis=2) < 200
    rows, cols = np.flatnonzero(ink.any(axis=1)), np.flatnonzero(ink.any(axis=0))
    if rows.size == 0:
        return []
    margin = 12
    left, right = max(x0, x0 + cols[0] - margin), min(x1, x0 + cols[-1] + margin)
    top, bottom = max(y0, y0 + rows[0] - margin), min(y1, y0 + rows[-1] + margin)
    lines: list[TextLine] = []
    for ty in _tile_starts(top, bottom):
        for tx in _tile_starts(left, right):
            tile = image[ty:min(ty + TILE, bottom), tx:min(tx + TILE, right)]
            if tile.min() > 200:
                continue
            x_end, y_end = tx + tile.shape[1], ty + tile.shape[0]
            for line in _ocr(tile, ZOOM, (tx, ty)):
                # A line touching an inner tile edge is cut; the next tile has it whole.
                bx0, by0, bx1, by1 = line.box
                cut = (
                    (bx0 < tx + 3 and tx > left) or (by0 < ty + 3 and ty > top)
                    or (bx1 > x_end - 3 and x_end < right) or (by1 > y_end - 3 and y_end < bottom)
                )
                if not cut:
                    lines.append(line)
    return _dedupe(lines)


def read_scale(image: np.ndarray, bottom: tuple[int, int, int, int]) -> int | None:
    """The scale stated in the title block (or, failing that, under the scale bar)."""
    x0, y0, x1, y1 = bottom
    middle = (x0 + x1) // 2
    for left, right in ((middle, x1), (x0, middle)):
        for line in _ocr(image[y0:y1, left:right], 1.0, (left, y0)):
            scale = parse_scale(line.text)
            if scale:
                return scale
    return None


@dataclass(frozen=True)
class _Target:
    """A patch of the plan to read again, and how much to enlarge it."""

    box: Box
    zoom: float


def _reread_targets(table: Table, numbers_only: bool = False) -> list[_Target]:
    """Places where a label is incomplete or a room number is missing."""
    targets: list[_Target] = []

    def add(box: Box, zoom: float) -> None:
        if all(_overlap_share(box, other.box) < 0.7 for other in targets):
            targets.append(_Target(box, zoom))

    def around(line: TextLine, up: float, down: float) -> Box:
        half = max(line.width / 2 + line.height, 45.0)
        return (line.cx - half, line.box[1] - up * line.height, line.cx + half, line.box[3] + down * line.height)

    for line in table.orphans:
        add(around(line, 2.2, 2.2), REREAD_ZOOM)
    for row in table.rows:
        if not row.certain:
            add(around(row.line, 1.6, 1.6), REREAD_ZOOM)
    prints_areas = table.prints_areas
    for row in table.rows:
        if numbers_only:
            break
        if row.name is None or (prints_areas and row.area is None):
            up = 1.8 if row.name is None else 0.3
            down = 1.8 if prints_areas and row.area is None else 0.3
            add(around(row.line, up, down), REREAD_ZOOM)
    targets = targets[:MAX_REREADS]

    # A missing number is usually next to the numbers just before and after it.
    by_index = {_split_number(row.number)[1]: row.line for row in table.rows}
    reach = float(np.clip(table.neighbour_spacing(), 70.0, 130.0))
    windows = 0
    for number in table.missing:
        k = _split_number(number)[1]
        for j in (k - 1, k + 1):
            if j in by_index and windows < MAX_GAP_WINDOWS:
                line = by_index[j]
                before = len(targets)
                add((line.cx - reach, line.cy - reach, line.cx + reach, line.cy + reach), GAP_ZOOM)
                windows += len(targets) - before
    return targets


def _ocr_patches(image: np.ndarray, targets: list[_Target], bold: bool = False) -> list[TextLine]:
    """Read many small patches in few OCR calls, by laying them out on shared canvases."""
    height, width = image.shape[:2]
    cells: list[tuple[np.ndarray, _Target, int, int]] = []
    for target in targets:
        x0, y0 = max(0, int(target.box[0])), max(0, int(target.box[1]))
        x1, y1 = min(width, int(target.box[2]) + 1), min(height, int(target.box[3]) + 1)
        if x1 - x0 < 8 or y1 - y0 < 8:
            continue
        patch = cv2.resize(
            image[y0:y1, x0:x1], None, fx=target.zoom, fy=target.zoom, interpolation=cv2.INTER_CUBIC
        )
        if bold:
            patch = _bold_only(patch, target.zoom)
        cells.append((patch, target, x0, y0))
    cells.sort(key=lambda cell: -cell[0].shape[0])

    # Shelf packing: fill a row left to right, start a new row below, then a new canvas.
    gap = 24
    canvases: list[list[tuple[int, int, int]]] = [[]]
    x = y = shelf = 0
    for index, (patch, _, _, _) in enumerate(cells):
        h, w = patch.shape[:2]
        if x and x + w > CANVAS:
            x, y, shelf = 0, y + shelf + gap, 0
        if y and y + h > CANVAS:
            canvases.append([])
            x = y = shelf = 0
        canvases[-1].append((index, x, y))
        x, shelf = x + w + gap, max(shelf, h)

    found: list[TextLine] = []
    for placed in canvases:
        if not placed:
            continue
        right = max(x + cells[i][0].shape[1] for i, x, _ in placed)
        bottom = max(y + cells[i][0].shape[0] for i, _, y in placed)
        canvas = np.full((bottom, right, 3), 255, np.uint8)
        for i, x, y in placed:
            patch = cells[i][0]
            canvas[y:y + patch.shape[0], x:x + patch.shape[1]] = patch
        for line in _ocr(canvas, 1.0, (0.0, 0.0)):
            for i, x, y in placed:
                patch, target, x0, y0 = cells[i]
                h, w = patch.shape[:2]
                if not (x <= line.cx < x + w and y <= line.cy < y + h):
                    continue
                # A line that touches the patch edge is cut off.
                if line.box[0] > x + 2 and line.box[1] > y + 2 and line.box[2] < x + w - 2 and line.box[3] < y + h - 2:
                    z = target.zoom
                    box = (
                        (line.box[0] - x) / z + x0, (line.box[1] - y) / z + y0,
                        (line.box[2] - x) / z + x0, (line.box[3] - y) / z + y0,
                    )
                    found.append(TextLine(line.text, box, line.confidence))
                break
    return found


def _value(line: TextLine, prefix: str | None) -> int:
    """How useful a line is to the table: decides which of two overlapping reads to keep."""
    text = line.text
    area = parse_area(text)
    if area and area[1]:
        return 3
    number = parse_number(text)
    if number and number.split(".")[0] == prefix:
        return 3
    if parse_tag(text) or snap_name(text, cutoff=0.9):
        return 3
    if area or number or snap_name(text):
        return 2
    return 1


def merge_rereads(lines: list[TextLine], rereads: list[TextLine], prefix: str | None) -> list[TextLine]:
    """Add second-pass lines to the first-pass ones; a more useful read replaces a worse one."""
    merged = list(lines)
    for new in sorted(rereads, key=lambda l: -l.confidence):
        rivals = [old for old in merged if _overlap_share(new.box, old.box) >= 0.5]
        new_value = _value(new, prefix)
        if all(new_value > _value(old, prefix) for old in rivals):
            merged = [old for old in merged if all(old is not rival for rival in rivals)]
            merged.append(new)
    return merged


def _tighten(image: np.ndarray, line: TextLine) -> TextLine:
    """Shrink a line's box to the ink inside it, plus a small margin."""
    height, width = image.shape[:2]
    x0, y0 = max(0, int(np.floor(line.box[0]))), max(0, int(np.floor(line.box[1])))
    x1, y1 = min(width, int(np.ceil(line.box[2]))), min(height, int(np.ceil(line.box[3])))
    if x1 <= x0 or y1 <= y0:
        return line
    ink = image[y0:y1, x0:x1].min(axis=2) < 160
    rows, cols = np.flatnonzero(ink.any(axis=1)), np.flatnonzero(ink.any(axis=0))
    if rows.size == 0:
        return line
    pad = 2.0
    box = (x0 + cols[0] - pad, y0 + rows[0] - pad, x0 + cols[-1] + 1 + pad, y0 + rows[-1] + 1 + pad)
    return replace(line, box=box)


def read_table(image: np.ndarray) -> tuple[Table, int | None]:
    """Read a plan into its label table and scale."""
    image = np.ascontiguousarray(image[..., :3])
    layout = find_layout(image)
    scale = read_scale(image, layout.bottom)
    clean = _without_red(image)
    lines = detect_lines(clean, layout.drawing)
    table = build_table(lines)
    # Each further look is at a different picture: what one misses the other often reads.
    view = _without_long_lines(clean)
    for bold in (False, True):
        targets = _reread_targets(table, numbers_only=bold)
        rereads = _ocr_patches(view, targets, bold)
        if rereads:
            lines = merge_rereads(lines, rereads, table.prefix)
            table = build_table(lines)
    # The detector's boxes are loose; the caller erases text by box and needs them snug.
    for row in table.rows:
        row.line = _tighten(view, row.line)
        row.name = _tighten(view, row.name) if row.name else None
        row.area = _tighten(view, row.area) if row.area else None
    table.tags = [(letter, _tighten(view, line)) for letter, line in table.tags]
    table.orphans = [_tighten(view, line) for line in table.orphans]
    return table, scale


def read_labels(image: np.ndarray) -> PlanLabels:
    """Read the room labels of a plan. ``image`` is an RGB uint8 array."""
    table, scale = read_table(image)
    return to_plan_labels(table, scale)
