"""Builds the floor-plan benchmark: the raster floor plans of a dataset that have questions."""

from __future__ import annotations

import json
import random
import shutil
from collections import Counter
from itertools import zip_longest
from pathlib import Path, PurePosixPath
from typing import Any

from chatchtg.documents.library import RASTER_SUFFIXES

# The dataset has no drawing-type field. These source folders hold floor plans only.
# To add a source, check that every questioned drawing in it is a floor plan.
FLOOR_PLAN_SOURCES = ("cubicasa5k", "swiss_dwellings")

QUESTIONS_FILE = "questions.jsonl"
SUBSET_FILE = "questions-subset.jsonl"
LICENCE_FILE = "LICENSE.md"

Question = dict[str, Any]


def _source(question: Question) -> str:
    return PurePosixPath(question["path"]).parts[1]


def _raster_path(root: Path, path: str) -> str | None:
    """Return the path of the drawing's raster file, or None when there is none.

    A drawing in another format is replaced by the PNG of the same name: the dataset
    states that files with the same name and a different extension are the same drawing.
    """
    drawing = PurePosixPath(path)
    if drawing.suffix.lower() not in RASTER_SUFFIXES:
        drawing = drawing.with_suffix(".png")
    target = (root / drawing).resolve()
    source_dir = (root / "data" / drawing.parts[1]).resolve()
    if not target.is_relative_to(source_dir) or not target.is_file():
        return None
    return drawing.as_posix()


def select_questions(source_dir: Path) -> tuple[list[Question], int]:
    """Return the questions about raster floor plans, and how many had no raster file.

    questions.jsonl is downloaded data, so a path that is absolute or contains ``..`` is
    ignored, like the questions of every other source.
    """
    root = source_dir.resolve()
    questions: list[Question] = []
    seen: set[tuple[str, str]] = set()
    dropped = 0
    with (root / QUESTIONS_FILE).open(encoding="utf-8") as lines:
        for line in lines:
            if not line.strip():
                continue
            question = json.loads(line)
            parts = PurePosixPath(question["path"]).parts
            if len(parts) < 3 or parts[0] != "data" or parts[1] not in FLOOR_PLAN_SOURCES or ".." in parts:
                continue
            path = _raster_path(root, question["path"])
            if path is None:
                dropped += 1
                continue
            key = (path, question["question"])
            if key not in seen:  # the same question on two formats of one drawing
                seen.add(key)
                questions.append({**question, "path": path})
    return questions, dropped


def _spread_over_drawings(positions: list[int], questions: list[Question], rng: random.Random) -> list[int]:
    """Order a stratum at random, so that each drawing is used once before any is used twice."""
    by_drawing: dict[str, list[int]] = {}
    for position in positions:
        by_drawing.setdefault(questions[position]["path"], []).append(position)
    drawings = list(by_drawing.values())
    rng.shuffle(drawings)
    for drawing in drawings:
        rng.shuffle(drawing)
    return [p for round_ in zip_longest(*drawings) for p in round_ if p is not None]


def sample_subset(questions: list[Question], size: int, seed: int) -> list[Question]:
    """Return ``size`` questions, balanced over (source, question type).

    The same questions, size and seed always give the same subset. A stratum with too few
    questions gives all it has, and the larger strata fill the rest. The subset keeps the
    order of ``questions``.
    """
    if size >= len(questions):
        return list(questions)
    rng = random.Random(seed)
    strata: dict[tuple[str, str], list[int]] = {}
    for position, question in enumerate(questions):
        strata.setdefault((_source(question), str(question["type"])), []).append(position)
    ordered = [_spread_over_drawings(strata[key], questions, rng) for key in sorted(strata)]
    picked = [p for round_ in zip_longest(*ordered) for p in round_ if p is not None][:size]
    return [questions[position] for position in sorted(picked)]


def _write_jsonl(path: Path, questions: list[Question]) -> None:
    with path.open("w", encoding="utf-8") as out:
        for question in questions:
            out.write(json.dumps(question, ensure_ascii=False) + "\n")


def _copy(source: Path, target: Path) -> bool:
    """Copy a file. Return False when the target is already there with the same size."""
    if target.is_file() and target.stat().st_size == source.stat().st_size:
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return True


def _readme(report: dict[str, Any], source_name: str, seed: int) -> str:
    sources = "\n".join(
        f"| {source} | {counts['images']} | {counts['questions']} |"
        for source, counts in report["by_source"].items()
    )
    types = ", ".join(f"{count} {question_type}" for question_type, count in report["by_type"].items())
    return f"""# Floor-plan benchmark

Raster floor plans with questions, taken from the `{source_name}` split of the Junction
construction-drawings dataset. Built with `chatchtg benchmark`; do not edit by hand.

```
data/<source>/<file>.png   the floor plans
data/<source>/{LICENCE_FILE}   licence and credit for every file in that folder
{QUESTIONS_FILE}            all questions: {{"path", "question", "answer", "unit", "type"}}
{SUBSET_FILE}     {report['subset']} of them, for routine runs (seed {seed})
```

{report['images']} floor plans, {report['questions']} questions ({types}).

| source | floor plans | questions |
|---|---|---|
{sources}

Only drawings that have at least one question are included. A question that pointed at
another format of a drawing (for example PDF) now points at the PNG of the same drawing.
The subset is balanced over source and question type, and uses each drawing once before
it uses any drawing twice.

Scoring: counts must be exact; areas, lengths and volumes within 2%.

## Licences

Every file keeps the licence in `data/<source>/{LICENCE_FILE}`. Keep the credit when you show or
share a drawing. Questions have the licence of the drawing they ask about.
"""


def build_benchmark(
    source_dir: Path, out_dir: Path, subset_size: int = 500, seed: int = 0
) -> dict[str, Any]:
    """Write the benchmark to ``out_dir`` and return its counts.

    ``out_dir`` gets the same layout as a downloaded dataset, with the images copied in.
    A file that is already there is not copied again.
    """
    source_root = source_dir.resolve()
    out_root = out_dir.resolve()
    if out_root == source_root or source_root.is_relative_to(out_root) or out_root.is_relative_to(source_root):
        raise ValueError(f"{out_dir} overlaps the source dataset {source_dir}")

    questions, dropped = select_questions(source_root)
    subset = sample_subset(questions, subset_size, seed)

    out_root.mkdir(parents=True, exist_ok=True)
    copied = 0
    images = sorted({question["path"] for question in questions})
    for path in images:
        copied += _copy(source_root / path, out_root / path)
    for source in sorted({_source(question) for question in questions}):
        licence = source_root / "data" / source / LICENCE_FILE
        if licence.is_file():
            _copy(licence, out_root / "data" / source / LICENCE_FILE)
    _write_jsonl(out_root / QUESTIONS_FILE, questions)
    _write_jsonl(out_root / SUBSET_FILE, subset)

    questions_by_source = Counter(_source(question) for question in questions)
    images_by_source = Counter(PurePosixPath(path).parts[1] for path in images)
    report: dict[str, Any] = {
        "questions": len(questions),
        "images": len(images),
        "dropped": dropped,
        "subset": len(subset),
        "copied": copied,
        "by_source": {
            source: {"questions": questions_by_source[source], "images": images_by_source[source]}
            for source in sorted(questions_by_source)
        },
        "by_type": dict(sorted(Counter(str(question["type"]) for question in questions).items())),
    }
    (out_root / "README.md").write_text(_readme(report, source_root.name, seed), encoding="utf-8")
    return report
