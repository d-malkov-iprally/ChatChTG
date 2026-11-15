"""Read every plan that has a room-count question, in parallel, to fill the OCR cache.

Usage: python -m chatchtg.rooms.warm <dataset_dir> <cache_dir> [workers]
"""

from __future__ import annotations

import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

from chatchtg.rooms import ocr
from chatchtg.rooms.questions import classify


def plans(dataset_dir: Path, questions_file: str = "questions.jsonl") -> dict[str, str]:
    """Return {path: profile} of the plans that have a room-count question."""
    found: dict[str, str] = {}
    with (dataset_dir / questions_file).open(encoding="utf-8") as lines:
        for line in lines:
            row = json.loads(line)
            if row["type"] == "count" and row["path"] not in found:
                room = classify(row["question"])
                if room is not None:
                    found[row["path"]] = room.profile
    return found


def _read(job: tuple[str, str, str, str]) -> str | None:
    dataset_dir, path, profile, cache_dir = job
    try:
        ocr.read_plan((Path(dataset_dir) / path).read_bytes(), profile, Path(cache_dir))
    except Exception as error:  # noqa: BLE001 - one bad file must not stop the run
        return f"{path}: {error!r}"
    return None


def main(argv: list[str]) -> int:
    dataset_dir, cache_dir = Path(argv[0]).resolve(), Path(argv[1]).resolve()
    workers = int(argv[2]) if len(argv) > 2 else 8
    jobs = [(str(dataset_dir), p, profile, str(cache_dir)) for p, profile in plans(dataset_dir).items()]
    jobs.sort(key=lambda job: job[2] != "swiss")
    start = time.time()
    errors = 0
    with Pool(workers) as pool:
        for done, error in enumerate(pool.imap_unordered(_read, jobs, chunksize=4), 1):
            if error:
                errors += 1
                print(error, flush=True)
            if done % 200 == 0:
                print(f"{done}/{len(jobs)} plans, {time.time() - start:.0f}s", flush=True)
    print(f"Done: {len(jobs)} plans, {errors} errors, {time.time() - start:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
