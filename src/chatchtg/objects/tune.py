"""Choose the minimum score of each detector class on the valid split of CubiCasa5K.

Usage: python -m chatchtg.objects.tune <questions.jsonl> <detections.jsonl> [split]

For each kind of object-count question, the minimum score that gives the most exact counts
against the benchmark's answers on the plans of ``split`` (default ``valid``); of
equally good values, the middle one. Showers and
bathtubs are counted together, so their two minimums are chosen together. Prints the values
for ``THRESHOLDS`` in ``counter.py``. Never tune on the test split: it is the split to report.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from chatchtg.objects.questions import KINDS, ObjectQuestion, classify

GRID = [round(0.05 + 0.01 * step, 2) for step in range(91)]  # 0.05 .. 0.95
PAIR_GRID = [round(0.05 + 0.05 * step, 2) for step in range(19)]


def answers(questions: Path) -> dict[str, dict[str, int]]:
    """Image name -> kind -> the benchmark's answer, for the CubiCasa5K object-count questions."""
    found: dict[str, dict[str, int]] = defaultdict(dict)
    with questions.open(encoding="utf-8") as lines:
        for line in lines:
            row = json.loads(line)
            question = classify(row["question"])
            if question is not None and "/cubicasa5k/" in f"/{row['path']}":
                found[row["path"].rsplit("/", 1)[-1]][question.kind] = int(row["answer"])
    return found


def scores(detections: Path, split: str) -> dict[str, dict[str, list[float]]]:
    """Image name -> class -> the scores of its detections, for the plans of ``split``.

    A plan whose image is also a plan of the train split is left out: the detector saw it.
    """
    found: dict[str, dict[str, list[float]]] = {}
    with detections.open(encoding="utf-8") as lines:
        rows = [json.loads(line) for line in lines]
    trained = {row["sha256"] for row in rows if row["split"] == "train"}
    for row in rows:
        if row["split"] == split and (split == "train" or row["sha256"] not in trained):
            per_class: dict[str, list[float]] = defaultdict(list)
            for detection in row["detections"]:
                per_class[detection["label"]].append(detection["score"])
            found[row["image"]] = per_class
    return found


def exact(
    plans: list[tuple[dict[str, list[float]], int]], classes: tuple[str, ...], minimums: dict[str, float]
) -> int:
    """On how many plans the count with these minimums is the answer."""
    return sum(
        sum(sum(s >= minimums[c] for s in per_class.get(c, [])) for c in classes) == answer
        for per_class, answer in plans
    )


def tune(
    answers_by_image: dict[str, dict[str, int]], scores_by_image: dict[str, dict[str, list[float]]]
) -> dict[str, dict]:
    """Kind -> the chosen minimums, and the exact counts with them and with 0.5."""
    result = {}
    for kind in KINDS:
        classes = ObjectQuestion(kind).classes
        plans = [
            (scores_by_image[image], kinds[kind])
            for image, kinds in answers_by_image.items()
            if kind in kinds and image in scores_by_image
        ]
        if len(classes) == 1:
            candidates = [{classes[0]: t} for t in GRID]
        else:
            candidates = [{classes[0]: a, classes[1]: b} for a in PAIR_GRID for b in PAIR_GRID]
        hits = [exact(plans, classes, m) for m in candidates]
        # Of the minimums that tie for the most exact counts, the middle one: it is the
        # farthest from the edges of the best range, where the count changes.
        tied = [m for m, h in zip(candidates, hits) if h == max(hits)]
        best = tied[len(tied) // 2]
        result[kind] = {
            "minimums": best,
            "plans": len(plans),
            "exact": exact(plans, classes, best),
            "exact_at_0.5": exact(plans, classes, dict.fromkeys(classes, 0.5)),
        }
    return result


def main(argv: list[str]) -> int:
    split = argv[2] if len(argv) > 2 else "valid"
    if split == "test":
        print("Refused: the test split is the split to report, not to tune on.", file=sys.stderr)
        return 2
    result = tune(answers(Path(argv[0])), scores(Path(argv[1]), split))
    print(f"{'kind':<13}{'plans':>6}{'exact':>7}{'rate':>8}{'at 0.5':>8}  minimums")
    for kind, row in result.items():
        print(
            f"{kind:<13}{row['plans']:>6}{row['exact']:>7}{row['exact'] / row['plans']:>8.1%}"
            f"{row['exact_at_0.5'] / row['plans']:>8.1%}  {row['minimums']}"
        )
    thresholds = {c: t for row in result.values() for c, t in row["minimums"].items()}
    print("THRESHOLDS =", json.dumps(thresholds))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
