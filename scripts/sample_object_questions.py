"""Draw object-count questions on plans that the detector never saw, for a three-way comparison.

Usage:
    uv run python scripts/sample_object_questions.py <questions.jsonl> <out.jsonl> [--per-kind 30] [--seed 0]

Takes the English questions of the five CubiCasa5K object-count templates (``objects/questions``)
on plans of CubiCasa5K's test split, as ``objects.report`` assigns them (a test plan whose image
is also a train plan counts as train, so it is left out). It shuffles the test plans with
``--seed`` and gives each kind, in turn, the next plan that has a question of that kind and
was not used yet, until each kind has ``--per-kind`` questions: one question per plan while
there are plans enough. The output is ordered round-robin over the kinds (window, door,
toilet, sink, bath_shower, window, ...), so that a run stopped by its cost limit still covers
the kinds evenly.

Needs the detections in the extraction store (``python -m chatchtg.objects.importer``): the
split of each plan is read from there. No LLM call.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from chatchtg.config import Settings
from chatchtg.objects.questions import KINDS, classify
from chatchtg.objects.report import by_source, split_of
from chatchtg.wiring import build_extraction_store


def english(question: str) -> bool:
    return question.lower().startswith("how many")


def draw(rows: list[dict], splits: dict[str, str], per_kind: int, seed: int) -> list[dict]:
    """``splits``: benchmark path -> split. Returns the chosen rows, round-robin over the kinds."""
    by_plan: dict[str, dict[str, dict]] = {}
    for row in rows:
        question = classify(row["question"])
        if question is None or not english(row["question"]) or splits.get(row["path"]) != "test":
            continue
        by_plan.setdefault(row["path"], {})[question.kind] = row
    plans = sorted(by_plan)
    random.Random(seed).shuffle(plans)
    chosen: dict[str, list[dict]] = {kind: [] for kind in KINDS}
    used: set[str] = set()
    # One plan per question first; reuse plans only if a kind runs out of fresh ones.
    for allow_reuse in (False, True):
        for kind in KINDS:
            for plan in plans:
                if len(chosen[kind]) >= per_kind:
                    break
                if kind not in by_plan[plan] or (plan in used and not allow_reuse):
                    continue
                if any(row["path"] == plan for row in chosen[kind]):
                    continue
                chosen[kind].append(by_plan[plan][kind])
                used.add(plan)
    ordered = []
    for index in range(per_kind):
        for kind in KINDS:
            if index < len(chosen[kind]):
                ordered.append(chosen[kind][index])
    return ordered


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("questions", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--per-kind", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    extractions = by_source(build_extraction_store(Settings.from_env()))
    splits = {path: split_of(extraction, path) for path, extraction in extractions.items()}
    with args.questions.open(encoding="utf-8") as lines:
        rows = [json.loads(line) for line in lines if line.strip()]
    chosen = draw(rows, splits, args.per_kind, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as out:
        for row in chosen:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    plans = {row["path"] for row in chosen}
    print(f"{len(chosen)} questions on {len(plans)} plans, seed {args.seed}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
