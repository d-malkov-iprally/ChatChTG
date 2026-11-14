"""Check the area-question rules and the image-based room membership on every plan.

Reads the questions of type "area" and the DXF room truth (var/room-truth, see
chatchtg.evals.room_truth) and reports:

(a) the share of questions the parser reads;
(b) the share of answers within 2% when the rules get the true areas and the true
    apartments, with outdoor rooms told from the image;
(c) the same with the apartments told from the image, and how many rooms get the right
    apartment.

    uv run python scripts/eval_membership.py [--jitter 6] [--workers 8] [--worst 15]

--jitter moves every label and tag by up to that many pixels, as OCR positions would be.
"""

from __future__ import annotations

import argparse
import json
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from chatchtg.areas.membership import assign_apartments, outdoor_rooms
from chatchtg.areas.questions import AreaQuery, RoomRow, answer_query, parse_question
from chatchtg.evals.room_truth import TruthPlan

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = ROOT / "data/floorplans/questions.jsonl"
IMAGES = ROOT / "data/floorplans/data/swiss_dwellings"
TRUTH = ROOT / "var/room-truth"
TOLERANCE = 0.02
KINDS = ("room", "label", "apartment", "floor")


@dataclass
class PlanResult:
    stem: str
    asked: Counter = field(default_factory=Counter)
    right_truth: Counter = field(default_factory=Counter)
    right_image: Counter = field(default_factory=Counter)
    rooms: int = 0
    rooms_wrong: int = 0
    misses: list[str] = field(default_factory=list)


def _within(value: float | None, reference: float) -> bool:
    return value is not None and abs(value - reference) <= TOLERANCE * abs(reference)


def evaluate_plan(job: tuple[str, list[tuple[AreaQuery, float]], float]) -> PlanResult:
    stem, questions, jitter = job
    plan = TruthPlan.from_json(json.loads((TRUTH / f"{stem}.json").read_text()))
    image = cv2.cvtColor(cv2.imread(str(IMAGES / f"{stem}.png")), cv2.COLOR_BGR2RGB)
    rng = np.random.default_rng(zlib.crc32(stem.encode()))

    def moved(point: tuple[float, float]) -> tuple[float, float]:
        dx, dy = rng.uniform(-jitter, jitter, 2) if jitter else (0.0, 0.0)
        return (point[0] + dx, point[1] + dy)

    points = [moved(room.label_px) for room in plan.rooms]
    tags = {letter: moved(point) for letter, point in plan.tags_px.items()}
    outdoor = outdoor_rooms(image, points)
    apartments = assign_apartments(image, points, tags)

    def table(letters: list[str | None]) -> list[RoomRow]:
        return [
            RoomRow(room.number, room.name, room.area, letter, shaded)
            for room, letter, shaded in zip(plan.rooms, letters, outdoor)
        ]

    by_truth = table([room.apartment for room in plan.rooms])
    by_image = table(apartments)
    result = PlanResult(stem=stem, rooms=len(plan.rooms))
    result.rooms_wrong = sum(a.apartment != b.apartment for a, b in zip(by_truth, by_image))
    for query, reference in questions:
        result.asked[query.kind] += 1
        for rows, right in ((by_truth, result.right_truth), (by_image, result.right_image)):
            answer = answer_query(query, rows)
            if _within(answer.value, reference):
                right[query.kind] += 1
            elif rows is by_image or query.kind != "apartment":
                which = "image" if rows is by_image else "truth"
                target = query.number or query.label or query.apartment or ""
                got = "none" if answer.value is None else f"{answer.value:.2f}"
                result.misses.append(
                    f"{query.kind} {target} [{which}]: {got} vs {reference:.2f}"
                )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--jitter", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--worst", type=int, default=15)
    parser.add_argument("--limit", type=int, default=0, help="only the first N plans")
    args = parser.parse_args()

    per_plan: dict[str, list[tuple[AreaQuery, float]]] = defaultdict(list)
    total = unparsed = 0
    for line in QUESTIONS.read_text().splitlines():
        row = json.loads(line)
        if row["type"] != "area":
            continue
        total += 1
        query = parse_question(row["question"])
        if query is None:
            unparsed += 1
            continue
        per_plan[Path(row["path"]).stem].append((query, float(row["answer"])))
    print(f"(a) parsed {total - unparsed} of {total} area questions on {len(per_plan)} plans")

    stems = sorted(per_plan)[: args.limit or None]
    missing = [stem for stem in stems if not (TRUTH / f"{stem}.json").exists()]
    if missing:
        print(f"no room truth for {len(missing)} plans, skipped")
    jobs = [(stem, per_plan[stem], args.jitter) for stem in stems if stem not in missing]
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(evaluate_plan, jobs, chunksize=8))

    asked, by_truth, by_image = Counter(), Counter(), Counter()
    for result in results:
        asked += result.asked
        by_truth += result.right_truth
        by_image += result.right_image
    print(f"jitter of label and tag positions: {args.jitter:g} px")
    print("share of answers within 2%, true areas, outdoor from the image:")
    print(f"  {'kind':10s} {'n':>6s}  (b) true apartment   (c) apartment from image")
    for kind in KINDS:
        n = asked[kind]
        if n:
            print(
                f"  {kind:10s} {n:6d}  {by_truth[kind] / n:8.2%} ({n - by_truth[kind]:3d} wrong)"
                f"   {by_image[kind] / n:8.2%} ({n - by_image[kind]:3d} wrong)"
            )
    n = sum(asked.values())
    print(f"  {'all':10s} {n:6d}  {sum(by_truth.values()) / n:8.2%}"
          f"               {sum(by_image.values()) / n:8.2%}")
    rooms = sum(r.rooms for r in results)
    wrong = sum(r.rooms_wrong for r in results)
    bad_plans = sum(r.rooms_wrong > 0 for r in results)
    print(f"rooms with the right apartment: {rooms - wrong} of {rooms} ({1 - wrong / rooms:.2%}); "
          f"{bad_plans} plans have a wrong room")

    print(f"\nworst {args.worst} plans (wrong rooms, wrong answers):")
    results.sort(key=lambda r: (-r.rooms_wrong, -len(r.misses), r.stem))
    for result in results[: args.worst]:
        if not result.rooms_wrong and not result.misses:
            break
        print(f"  {result.stem}: {result.rooms_wrong} of {result.rooms} rooms, "
              f"{len(result.misses)} answers")
        for miss in result.misses[:6]:
            print(f"      {miss}")


if __name__ == "__main__":
    main()
