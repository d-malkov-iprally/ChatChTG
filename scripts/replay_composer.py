"""Score a composer by replay: reuse answers from earlier eval runs whose context is the same.

    uv run python scripts/replay_composer.py data/floorplans --composer combined \\
        --questions questions-subset.jsonl --sample 100 --seed 0 \\
        --runs var/eval-runs/fix-issues-baseline-opus-5-5-100.jsonl var/eval-runs/fix-issues-areas-opus-5-5-100.jsonl \\
        --extra var/eval-runs/baseline-opus-5-5-100.jsonl var/eval-runs/room-counting/*.jsonl

No LLM is called. For each question of the sample, the composer builds its context, and the
script looks for a row of an earlier run with the same drawing, question and context (system
prompt, text parts, image hash and location request). The first of ``--runs`` that has one
gives the answer that is scored; ``--extra`` files are searched after them. A question with no
such row has no answer: the score is then a range, from all of them wrong to all of them right.

A context that carries an answer stated in code (``Context.computed.statement``) is sent to
no LLM. Its statement is scored, at no cost, and no earlier row is used: an earlier row with
the same text parts holds the answer of an LLM.

A replay is not a fresh run: the answers were made by the model at another time, and its
answers are not deterministic. The area pipeline runs (OCR, seconds for each new drawing).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from chatchtg.context.composer import ComposerInput, context_composer
from chatchtg.context.registry import get_composer
from chatchtg.documents.metadata import DocumentRecord
from chatchtg.evals.benchmark import sample_subset
from chatchtg.evals.runner import read_questions
from chatchtg.evals.scoring import score

STATED = "stated in code"

MEDIA_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as lines:
        return [json.loads(line) for line in lines if line.strip()]


def _key(path: str, question: str, context: dict[str, Any]) -> tuple[str, str, str]:
    """What the LLM gets for a question. Rows of older runs have no ``locations``."""
    return (path, question, json.dumps([context["system"], context["parts"], context.get("locations")]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--composer", default="combined")
    parser.add_argument("--questions", default="questions-subset.jsonl")
    parser.add_argument("--sample", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--extra", type=Path, nargs="*", default=[])
    parser.add_argument("--out", type=Path, help="one JSON line per question, with the source of its answer")
    args = parser.parse_args()

    answers: dict[tuple[str, str, str], list[tuple[Path, dict[str, Any]]]] = defaultdict(list)
    for path in [*args.runs, *args.extra]:
        for row in _rows(path):
            if "context" in row:
                answers[_key(row["path"], row["question"], row["context"])].append((path, row))

    composer = get_composer(args.composer)
    questions = list(read_questions(args.dataset_dir, args.questions))
    if args.sample is not None:
        questions = sample_subset(questions, args.sample, args.seed)
    results = []
    for question in questions:
        drawing = args.dataset_dir / question["path"]
        content = drawing.read_bytes()
        document = DocumentRecord(
            id="-", filename=drawing.name, media_type=MEDIA_TYPES[drawing.suffix.lower()],
            size_bytes=len(content), sha256="-", location=str(drawing), created_at=datetime.now(),
        )
        request = ComposerInput(document=document, content=content, message=question["question"])
        composed = composer.compose(request)
        context = composed.summary()
        part = context_composer(composer, request)
        statement = composed.computed.statement if composed.computed is not None else None
        found = [] if statement else answers.get(_key(question["path"], question["question"], context), [])
        source, row = found[0] if found else (None, None)
        if statement:
            row = {
                "composer": f"{(part or composer).name}@{(part or composer).version}",
                "correct": score(question["type"], question["answer"], statement),
                "response": statement,
                "cost_usd": 0.0,
            }
        results.append(
            {
                "path": question["path"],
                "question": question["question"],
                "type": question["type"],
                "expected": question["answer"],
                "part": part.name if part else composer.name,
                "adds_text": [p["text"] for p in context["parts"][1:-1] if p["type"] == "text"],
                "source": STATED if statement else source.name if source else None,
                "composer_of_source": row["composer"] if row else None,
                "correct": row["correct"] if row else None,
                "response": row["response"] if row else None,
                "cost_usd": row.get("cost_usd") if row else None,
                "all_sources": sorted({(p.name, r["correct"]) for p, r in found}),
            }
        )

    if args.out:
        args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results))
    by_source: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for result in results:
        counts = by_source[result["source"] or "no answer"]
        counts[0] += 1
        counts[1] += bool(result["correct"])
    missing = [r for r in results if r["source"] is None]
    right = sum(bool(r["correct"]) for r in results)
    print(f"Composer: {args.composer}. Questions: {len(results)}.")
    for source, (total, correct) in sorted(by_source.items()):
        print(f"  {source:<48}{total:>5} questions{correct:>5} right")
    print(f"Right: {right} to {right + len(missing)} of {len(results)} ({len(missing)} without an answer).")
    for result in missing:
        print(f"  no answer: [{result['part']}] {result['path']} | {result['question']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
