"""Score the label reader against the DXF ground truth on a random sample of plans.

    uv run python scripts/eval_labels.py 30 30 --seed 1
    uv run python scripts/eval_labels.py 32 32 --seed 202 --exclude-seed 101   # held out

The two numbers are how many 1:100 and 1:200 plans to draw. ``--exclude-seed`` leaves out
the plans that a 20 + 20 draw with that seed gives (the development sample). Needs
``data/`` and the truth files written by ``chatchtg.evals.room_truth`` in
``var/room-truth``.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from chatchtg.areas import labels as label_reader
from chatchtg.areas.labels import PlanLabels, fold, read_labels
from chatchtg.evals.room_truth import TruthPlan

ROOT = Path(__file__).resolve().parents[1]
PLANS = ROOT / "data/floorplans/data/swiss_dwellings"
TRUTH = ROOT / "var/room-truth"
MATCH_PX = 40.0

Counts = dict[str, list[int]]


def load_truth(stem: str) -> TruthPlan:
    return TruthPlan.from_json(json.loads((TRUTH / f"{stem}.json").read_text()))


def load_image(stem: str) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(str(PLANS / f"{stem}.png")), cv2.COLOR_BGR2RGB)


def sample(count_100: int, count_200: int, seed: int) -> list[str]:
    by_scale: dict[int, list[str]] = defaultdict(list)
    for path in sorted(TRUTH.glob("*.json")):
        by_scale[json.loads(path.read_text())["scale"]].append(path.stem)
    rng = random.Random(seed)
    return rng.sample(by_scale[100], count_100) + rng.sample(by_scale[200], count_200)


def _near(a: tuple[float, float], b: tuple[float, float], limit: float = MATCH_PX) -> bool:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 <= limit**2


def score(truth: TruthPlan, labels: PlanLabels) -> tuple[Counts, list[str]]:
    """Counts of [hits, trials] per metric, and one line per mistake."""
    counts: Counts = defaultdict(lambda: [0, 0])
    notes: list[str] = []

    def add(metric: str, hit: bool) -> None:
        counts[metric][0] += bool(hit)
        counts[metric][1] += 1

    add("scale", labels.scale == truth.scale)
    prints_areas = any(room.printed_area is not None for room in truth.rooms)
    matched: set[int] = set()
    for room in truth.rooms:
        found = None
        for index, label in enumerate(labels.rooms):
            if label.number == room.number and _near(label.number_xy, room.label_px):
                found = index
                break
        add("number recall", found is not None)
        if found is None:
            notes.append(f"missed {room.number} {room.name} at {room.label_px[0]:.0f},{room.label_px[1]:.0f}")
            continue
        matched.add(found)
        label = labels.rooms[found]
        if room.name is None:
            add("no-name kept", label.name is None)
        else:
            add("name exact", label.name is not None and fold(label.name) == fold(room.name))
        if fold(label.name or "") != fold(room.name or ""):
            notes.append(f"name {room.number}: read {label.name!r}, truth {room.name!r}")
        if prints_areas:
            if room.printed_area is not None:
                add("printed area exact", label.printed_area == room.printed_area)
            else:
                add("no-area kept", label.printed_area is None)
            if label.printed_area != room.printed_area:
                notes.append(f"area {room.number}: read {label.printed_area}, truth {room.printed_area}")
        elif label.printed_area is not None:
            notes.append(f"area {room.number}: read {label.printed_area}, none printed")
        polygon = np.array(room.polygon_px, np.float32)
        inside = cv2.pointPolygonTest(polygon, (float(label.seed[0]), float(label.seed[1])), False) > 0
        add("seed inside room", inside)
        if not inside:
            notes.append(f"seed of {room.number} outside its room")
    for index, label in enumerate(labels.rooms):
        add("number precision", index in matched)
        if index not in matched:
            notes.append(f"spurious {label.number} at {label.number_xy[0]:.0f},{label.number_xy[1]:.0f}")
    for letter, position in truth.tags_px.items():
        add("tag recall", letter in labels.tags and _near(labels.tags[letter], position))
    for letter in labels.tags:
        add("tag precision", letter in truth.tags_px and _near(labels.tags[letter], truth.tags_px[letter]))
    truth_numbers = {room.number for room in truth.rooms}
    found_numbers = {label.number for label in labels.rooms}
    add("plan: all rooms found", truth_numbers <= found_numbers and len(matched) == len(truth.rooms))
    add("plan: flags its gaps", truth_numbers <= found_numbers | set(labels.missing_numbers))
    # Text boxes: every truth label text should be mostly covered by a box of the reader.
    for box in truth.text_boxes:
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        add("text box covered", any(b[0] <= cx <= b[2] and b[1] <= cy <= b[3] for b in labels.text_boxes))
    return counts, notes


def run_plan(stem: str) -> tuple[str, int, Counts, list[str], float]:
    truth = load_truth(stem)
    image = load_image(stem)
    start = time.perf_counter()
    labels = read_labels(image)
    seconds = time.perf_counter() - start
    counts, notes = score(truth, labels)
    return stem, truth.scale, dict(counts), notes, seconds


def _limit_threads(threads: int) -> None:
    label_reader.OCR_THREADS = threads


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("count_100", type=int, help="number of 1:100 plans")
    parser.add_argument("count_200", type=int, help="number of 1:200 plans")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exclude-seed", type=int, help="leave out the 20 + 20 draw of this seed")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--threads", type=int, default=2, help="OCR threads per worker")
    parser.add_argument("--verbose", action="store_true", help="list every mistake")
    args = parser.parse_args()

    stems = sample(args.count_100, args.count_200, args.seed)
    if args.exclude_seed is not None:
        excluded = set(sample(20, 20, args.exclude_seed))
        stems = [stem for stem in stems if stem not in excluded]
    totals: dict[int, Counts] = {100: defaultdict(lambda: [0, 0]), 200: defaultdict(lambda: [0, 0])}
    seconds: dict[int, list[float]] = {100: [], 200: []}
    start = time.perf_counter()
    if args.workers > 1:
        with ProcessPoolExecutor(args.workers, initializer=_limit_threads, initargs=(args.threads,)) as pool:
            results = list(pool.map(run_plan, stems))
    else:
        results = [run_plan(stem) for stem in stems]
    wall = time.perf_counter() - start
    for stem, scale, counts, notes, took in results:
        seconds[scale].append(took)
        for metric, (hits, trials) in counts.items():
            totals[scale][metric][0] += hits
            totals[scale][metric][1] += trials
        if args.verbose and notes:
            print(f"{stem} (1:{scale}, {took:.1f} s)")
            for note in notes:
                print(f"    {note}")

    metrics = list(dict.fromkeys(m for scale in (100, 200) for m in totals[scale]))
    print(f"{'metric':24s} {'1:100':>20s} {'1:200':>20s}")
    print(f"{'plans':24s} {len(seconds[100]):>20d} {len(seconds[200]):>20d}")
    for metric in metrics:
        cells = []
        for scale in (100, 200):
            hits, trials = totals[scale][metric]
            cells.append(f"{hits}/{trials} {100 * hits / trials:5.1f}%" if trials else "-")
        print(f"{metric:24s} {cells[0]:>20s} {cells[1]:>20s}")
    cells = [f"{np.mean(seconds[s]):.2f}" if seconds[s] else "-" for s in (100, 200)]
    print(f"{'seconds per plan':24s} {cells[0]:>20s} {cells[1]:>20s}")
    print(f"wall time {wall:.0f} s with {args.workers} worker(s); seconds per plan are per worker")
    if args.workers > 1:
        print("(workers share the cores: use --workers 1 for the speed of a single reader)")


if __name__ == "__main__":
    main()
