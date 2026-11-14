"""Score the room measurer alone against the DXF ground truth, with no OCR and no LLM call.

    uv run python scripts/eval_measure.py --stems docs/benchmarks/stems/tune-200.txt
    uv run python scripts/eval_measure.py --stems FILE --out NEW.jsonl --compare OLD.jsonl

The label positions and text boxes come from the DXF truth (var/room-truth, built by
scripts/build_room_truth.py), so every error is an error of the measurer. A room counts as
right when its area is within 2% of the reference. Each wrong room gets a failure pattern
(chatchtg.evals.measure_score). ``--compare`` lists the rooms that changed against an
earlier run. The plans are read from CHATCHTG_DATA_ROOT (default ``data``).
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from chatchtg.areas.measure import measure_rooms
from chatchtg.config import Settings
from chatchtg.evals.measure_score import edge_offsets, failure_pattern
from chatchtg.evals.room_truth import TruthPlan

SETTINGS = Settings.from_env()
PLANS = SETTINGS.data_root / "floorplans" / "data" / "swiss_dwellings"
TRUTH = SETTINGS.state_dir / "room-truth"
PATTERNS = ("ok", "lost", "both", "leak", "edges", "no-area")
GROUPS = {
    "bath": ("bad", "bain", "bagno"),
    "kitchen": ("küche", "cuisine", "cucina"),
}
SIZES = ((float("-inf"), 4.0, "under 4 m²"), (4.0, 8.0, "4 to 8 m²"), (8.0, 15.0, "8 to 15 m²"), (15.0, float("inf"), "over 15 m²"))


def group_of(name: str | None) -> str:
    if not name:
        return "no name"
    folded = name.casefold()
    for group, prefixes in GROUPS.items():
        if folded.startswith(prefixes):
            return group
    return "other"


def size_of(area: float) -> str:
    return next(label for low, high, label in SIZES if low <= area < high)


def measure_plan(stem: str) -> list[dict]:
    plan = TruthPlan.from_json(json.loads((TRUTH / f"{stem}.json").read_text()))
    image = np.array(Image.open(PLANS / f"{stem}.png").convert("RGB"))
    seeds = [room.label_px for room in plan.rooms]
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    measures = measure_rooms(image, seeds, plan.pixels_per_metre, plan.text_boxes)
    seconds, cpu_seconds = time.perf_counter() - wall_start, time.process_time() - cpu_start
    rows = []
    for room, measure in zip(plan.rooms, measures):
        rows.append(
            {
                "stem": stem, "scale": plan.scale, "number": room.number, "name": room.name,
                "truth": room.area, "area": measure.area_m2, "pixel_area": measure.pixel_area_m2,
                "flags": measure.flags,
                "pattern": failure_pattern(room.polygon_px, room.area, measure.polygon, measure.area_m2),
                "offsets": [[round(length, 1), round(offset, 2)] for length, offset in edge_offsets(room.polygon_px, measure.polygon)],
                "seconds": round(seconds, 3), "cpu_seconds": round(cpu_seconds, 3),
            }
        )
    return rows


def share(rows: list[dict]) -> str:
    if not rows:
        return "-"
    ok = sum(row["pattern"] == "ok" for row in rows)
    return f"{ok / len(rows):6.1%} ({ok}/{len(rows)})"


def report(rows: list[dict]) -> None:
    by_scale: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_scale[row["scale"]].append(row)
    for scale in sorted(by_scale):
        part = by_scale[scale]
        plans = {row["stem"]: (row["seconds"], row["cpu_seconds"]) for row in part}
        patterns = Counter(row["pattern"] for row in part)
        print(f"1:{scale}: {len(plans)} plans, {len(part)} rooms, within 2%: {share(part)}")
        print("  patterns: " + ", ".join(f"{name} {patterns[name]}" for name in PATTERNS))
        for name in ("bath", "kitchen", "no name", "other"):
            print(f"  {name:12s} {share([row for row in part if group_of(row['name']) == name])}")
        for _, _, label in SIZES:
            print(f"  {label:12s} {share([row for row in part if size_of(row['truth']) == label])}")
        print(
            f"  seconds per plan: {statistics.mean(s for s, _ in plans.values()):.2f} elapsed, "
            f"{statistics.mean(c for _, c in plans.values()):.2f} CPU"
        )


def compare(rows: list[dict], old_path: Path) -> None:
    old = {}
    for line in old_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        old[(row["stem"], row["number"])] = row
    pairs = [(old[(row["stem"], row["number"])], row) for row in rows if (row["stem"], row["number"]) in old]
    print(f"compared with {old_path}: {len(pairs)} rooms in both runs")
    for scale in sorted({new["scale"] for _, new in pairs}):
        part = [(before, new) for before, new in pairs if new["scale"] == scale]
        changed = sum(before["area"] != new["area"] for before, new in part)
        worse = [(b, n) for b, n in part if b["pattern"] == "ok" and n["pattern"] != "ok"]
        better = [(b, n) for b, n in part if b["pattern"] != "ok" and n["pattern"] == "ok"]
        print(f"1:{scale}: area changed for {changed} rooms; ok -> not ok: {len(worse)}; not ok -> ok: {len(better)}")
        for title, group in (("ok -> not ok", worse), ("not ok -> ok", better)):
            for before, new in group[:20]:
                print(
                    f"  {title}: {new['stem']} {new['number']} {new['name'] or '-'}: truth {new['truth']:.2f}, "
                    f"{before['area']} ({before['pattern']}) -> {new['area']} ({new['pattern']})"
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stems", type=Path, required=True, help="file with one plan stem per line")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out", type=Path, default=Path("var/eval-measure.jsonl"))
    parser.add_argument("--compare", type=Path, help="the --out file of an earlier run on the same plans")
    args = parser.parse_args()

    stems = args.stems.read_text().split()
    with ProcessPoolExecutor(args.workers) as pool:
        rows = [row for plan_rows in pool.map(measure_plan, stems) for row in plan_rows]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")
    report(rows)
    if args.compare:
        compare(rows, args.compare)


if __name__ == "__main__":
    main()
