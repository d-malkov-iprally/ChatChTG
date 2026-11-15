"""Count where the combined composer sends each question of a question file. No LLM, no OCR.

    uv run python scripts/route_questions.py data/floorplans/questions.jsonl

Paths:
- ``areas``: the area parser reads the question (``areas/questions.py``).
- ``room-conventions``: a room count on a Finnish plan (``rooms/questions.py``); the context
  carries the counting conventions.
- ``baseline``: every other question. It goes to ``room-conventions`` too, which then builds
  the baseline context.

Fails when a question is claimed by both the area parser and the room classifier. Prints the
questions that reach the baseline path, grouped by source, type and opening words, so that an
area question or a Finnish room count that the parsers miss can be seen.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from chatchtg.areas.questions import parse_question
from chatchtg.context.composer import ComposerInput
from chatchtg.context.registry import get_composer
from chatchtg.context.room_conventions import conventions
from chatchtg.documents.metadata import DocumentRecord
from chatchtg.rooms.questions import classify

COMBINED = get_composer("combined")
# The choice reads only the message, so one stand-in drawing serves for every question.
_DOCUMENT = DocumentRecord(
    id="-", filename="plan.png", media_type="image/png", size_bytes=0, sha256="-",
    location="-", created_at=datetime(2026, 1, 1),
)


def route(message: str) -> str:
    """The part the combined composer picks, and ``baseline`` when that part is the room
    conventions composer and it adds no conventions."""
    chosen = COMBINED.delegate(ComposerInput(document=_DOCUMENT, content=b"", message=message))
    if chosen.name == "room-conventions" and conventions(message) is None:
        return "baseline"
    return chosen.name


def main(path: Path) -> int:
    paths: Counter[str] = Counter()
    by_source: Counter[tuple[str, str, str]] = Counter()
    unrouted: Counter[tuple[str, str, str]] = Counter()
    both = []
    with path.open(encoding="utf-8") as lines:
        for line in lines:
            if not line.strip():
                continue
            question = json.loads(line)
            message = question["question"]
            source = Path(question["path"]).parts[1]
            if parse_question(message) is not None and classify(message) is not None:
                both.append(message)
            chosen = route(message)
            paths[chosen] += 1
            by_source[(source, question["type"], chosen)] += 1
            if chosen == "baseline":
                opening = re.sub(r"(?:[«\"‘“]|'(?=[A-ZÄÖÜÉ])).*", "…", message)[:70]
                unrouted[(source, question["type"], opening)] += 1
    total = sum(paths.values())
    print(f"Questions: {total}")
    for name in ("areas", "room-conventions", "baseline"):
        print(f"  {name:<18}{paths[name]:>8}  {paths[name] / total:6.1%}")
    print("\nBy source, type and path:")
    for (source, kind, chosen), count in sorted(by_source.items()):
        print(f"  {source:<16}{kind:<7}{chosen:<18}{count:>8}")
    print("\nQuestions on the baseline path, by opening words:")
    for (source, kind, opening), count in sorted(unrouted.items()):
        print(f"  {count:>6}  {source:<16}{kind:<7}{opening}")
    if both:
        print(f"\n{len(both)} questions are claimed by both parsers, e.g. {both[0]!r}", file=sys.stderr)
        return 1
    print("\nNo question is claimed by both the area parser and the room classifier.")
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
