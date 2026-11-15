"""Measure every room-count question of a dataset and store the measurements.

Usage: python -m chatchtg.rooms.bench <dataset_dir> <cache_dir> <out.jsonl> [workers] [source]
``source`` keeps the plans whose path contains it, for example ``swiss_dwellings``.
Each output line is the question row with ``kind``, ``label`` and ``signals``.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

from chatchtg.rooms import router
from chatchtg.rooms.questions import RoomQuestion, classify


def _measure(job: tuple[str, str, list[tuple[str, str | None]], str]) -> tuple[str, dict]:
    dataset_dir, path, questions, cache_dir = job
    content = (Path(dataset_dir) / path).read_bytes()
    out = {}
    for kind, label in questions:
        try:
            out[f"{kind}|{label}"] = router.signals(content, RoomQuestion(kind, label), Path(cache_dir))
        except Exception as error:  # noqa: BLE001
            out[f"{kind}|{label}"] = {"error": repr(error)}
    return path, out


def main(argv: list[str]) -> int:
    dataset_dir, cache_dir, out_path = Path(argv[0]).resolve(), Path(argv[1]).resolve(), Path(argv[2])
    workers = int(argv[3]) if len(argv) > 3 else 8
    source = argv[4] if len(argv) > 4 else ""
    rows = []
    per_plan: dict[str, set[tuple[str, str | None]]] = defaultdict(set)
    with (dataset_dir / "questions.jsonl").open(encoding="utf-8") as lines:
        for line in lines:
            row = json.loads(line)
            room = classify(row["question"]) if row["type"] == "count" else None
            if room is not None and source in row["path"]:
                rows.append((row, room))
                per_plan[row["path"]].add((room.kind, room.label))
    jobs = [(str(dataset_dir), p, sorted(q, key=str), str(cache_dir)) for p, q in per_plan.items()]
    measured: dict[str, dict] = {}
    with Pool(workers) as pool:
        for done, (path, out) in enumerate(pool.imap_unordered(_measure, jobs, chunksize=4), 1):
            measured[path] = out
            if done % 500 == 0:
                print(f"{done}/{len(jobs)} plans", flush=True)
    with out_path.open("w", encoding="utf-8") as out_file:
        for row, room in rows:
            found = measured[row["path"]][f"{room.kind}|{room.label}"]
            record = {**row, "kind": room.kind, "label": room.label, "signals": found}
            out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} questions on {len(jobs)} plans to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
