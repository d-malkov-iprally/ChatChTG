"""Pick the counting method for each kind of room question.

``signals`` measures the drawing; ``decide`` turns the measurements into an answer. They are
separate so a policy can be scored on stored measurements without reading the drawings again.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from chatchtg.rooms import geometry, lexicon, ocr
from chatchtg.rooms.questions import RoomQuestion, classify

Signals = dict[str, Any]

# Words that show that the room labels of a Finnish plan were read at all.
_GATE = re.compile(
    r"^(MH\d*|OH|ET|KH|KPH|PH|PSH|VH|WC|K|KK|S|SAUNA|PARVEKE|ETEINEN|MAKUUH|OLOH|AUL|KEITTIÖ"
    r"|SOVRUM|KÖK|BADRUM|HALL|VARDAGSRUM|TK|KT)$"
)


@dataclass
class Finding:
    question: RoomQuestion
    count: int
    method: str
    """How the count was made: ``labels``, ``shapes``, ``usual`` (the usual answer) or a mix."""
    confident: bool
    """False when the measurements are weak or disagree; a caller may then ask the LLM."""
    evidence: list[dict[str, Any]] = field(default_factory=list)
    """What was counted: ``{"text", "box"}`` with the box in image pixels (x0, y0, x1, y1)."""
    signals: Signals = field(default_factory=dict)


def _evidence(tokens: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"text": t["text"], "box": [round(v) for v in t["box"]]} for t in tokens]


def signals(content: bytes, question: RoomQuestion, cache_dir: Path | None = None) -> Signals:
    """Measure what the question needs from the drawing."""
    read = ocr.read_plan(content, question.profile, cache_dir)
    kind = question.kind
    found: Signals = {"tokens": len(read["tokens"]), "mirrored": read.get("mirrored", False)}
    if question.profile == "finnish":
        seen = {w for t in read["tokens"] if t["conf"] >= 0.5 for w in lexicon.words(t["text"])}
        found["room_words"] = len([w for w in seen if _GATE.match(w)])
        typology = lexicon.typology(read)
        found["typology"] = typology["parts"] if typology else None
        found["typology_rooms"] = typology["rooms"] if typology else None
        count, tokens = lexicon.count_cubi_labels(read, kind)
        found["labels"] = count
        found["evidence"] = _evidence(tokens)
        if kind == "storeys":
            found["shapes"], boxes = geometry.count_storeys(geometry.decode(content))
            found["evidence"] = [
                {"text": "floor drawing", "box": [x, y, x + w, y + h]} for x, y, w, h in boxes
            ]
        if kind == "bathroom":
            found["wc_labels"] = lexicon.count_cubi_labels(read, "wc")[0]
    elif kind == "label":
        assert question.label is not None
        hits = lexicon.swiss_label_hits(read, question.label)
        found["labels"] = len(hits)
        found["labels_exact"] = len(lexicon.swiss_label_hits(read, question.label, fuzzy=False))
        found["evidence"] = _evidence(hits)
    else:
        count, tokens = lexicon.swiss_apartments_count(read)
        found["labels"] = count
        found["tags"] = len(tokens)
        found["shapes"], boxes = geometry.count_apartments(geometry.decode(content))
        found["kitchens"] = max(
            len(lexicon.swiss_label_hits(read, label)) for label in ("Küche", "Cuisine", "Cucina")
        )
        found["evidence"] = _evidence(tokens) or [
            {"text": "unit outline", "box": [x, y, x + w, y + h]} for x, y, w, h in boxes
        ]
    return found


def decide(kind: str, s: Signals) -> tuple[int, str, bool]:
    """Return (count, method, confident) for one kind of question.

    ``confident`` is True only where the count was measured to be as accurate as the LLM:
    apartments, when the unit tags and the unit outlines give the same number (exact on all
    1,970 such questions of the benchmark). Every other count is less accurate than the LLM
    (docs/benchmarks/2026-11-14-room-counting.md), so it is a fallback, not an answer to prefer.
    """
    labels = s["labels"]
    if kind == "label":
        return labels, "labels", False
    if kind == "apartments":
        shapes = s["shapes"]
        if labels and labels == shapes:
            return labels, "labels+shapes", True
        if labels:
            return labels, "labels", False
        return shapes, "shapes", False
    if kind == "storeys":
        return s["shapes"], "shapes", False
    if kind == "bathroom":
        return 1, "usual", False
    read_ok = s["room_words"] >= 2  # were the room labels of this plan read at all?
    if kind == "kitchen":
        # Nearly every home has one kitchen. Its label is often missed, and a second label
        # is usually the kitchen of a neighbouring home.
        return 1, "usual", False
    if kind == "sauna":
        if labels == 0 and "S" in (s.get("typology") or []):
            return 1, "labels", False
        return (labels, "labels", False) if read_ok else (1, "usual", False)
    if kind == "bedroom":
        return (labels, "labels", False) if read_ok else (1, "usual", False)
    # wc, balcony
    return (labels, "labels", False) if read_ok else (0, "usual", False)


def solve(content: bytes, message: str, cache_dir: Path | None = None) -> Finding | None:
    """Answer a room-count question from the drawing, or None when it is another question."""
    question = classify(message)
    if question is None:
        return None
    found = signals(content, question, cache_dir)
    count, method, confident = decide(question.kind, found)
    evidence = found.pop("evidence", [])
    return Finding(question, count, method, confident, evidence if method != "usual" else [], found)
