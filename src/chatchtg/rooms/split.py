"""Which plans a room-counting score may be reported on.

The lexicon rules were tuned on a fixed sample of plans (``tuned``). The other plans are
divided in two by a hash of their path: ``dev`` for further tuning, ``test`` for reporting.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from chatchtg.rooms.questions import classify

_TUNED = {"cubicasa5k": 600, "swiss_dwellings": 400}


def room_questions(dataset_dir: Path, questions_file: str = "questions.jsonl") -> list[dict]:
    """The room-count questions, each with ``kind`` and ``label`` added."""
    rows = []
    with (dataset_dir / questions_file).open(encoding="utf-8") as lines:
        for line in lines:
            row = json.loads(line)
            room = classify(row["question"]) if row["type"] == "count" else None
            if room is not None:
                rows.append({**row, "kind": room.kind, "label": room.label})
    return rows


def tuned_plans(rows: list[dict]) -> set[str]:
    """The plans that the prototype's rules were tuned on: its seeded sample of each source."""
    tuned: set[str] = set()
    for source, size in _TUNED.items():
        plans = sorted({row["path"] for row in rows if f"/{source}/" in row["path"]})
        random.Random(0).shuffle(plans)
        tuned.update(plans[:size])
    return tuned


def split_of(path: str, tuned: set[str]) -> str:
    if path in tuned:
        return "tuned"
    return "dev" if hashlib.sha1(path.encode()).digest()[0] % 2 == 0 else "test"
