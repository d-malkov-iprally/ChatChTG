"""Draw the evidence of an eval run on the drawings, for a review by eye.

    uv run python scripts/render_evidence.py RESULTS DATASET_DIR OUT_DIR [--every N] [--limit N] [--paths NAME ...]

RESULTS is a result file of ``chatchtg eval --evidence``. The script takes the rows whose
context asked the model for locations, in file order: every N-th of them, at most
``--limit``, and also every such row whose path ends with one of ``--paths``. For each it
writes ``OUT_DIR/<line number>-<file stem>.png`` with the marks drawn by
``chatchtg.evidence.render_evidence``, and prints one line.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from chatchtg.evidence import render_evidence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", type=Path)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--every", type=int, default=1, help="take every N-th row that asked for locations")
    parser.add_argument("--limit", type=int, help="take at most this many rows by --every")
    parser.add_argument("--paths", nargs="*", default=[], help="also take rows whose path ends with one of these")
    args = parser.parse_args(argv)
    if args.every < 1:
        parser.error("--every must be 1 or more")

    with args.results.open(encoding="utf-8") as lines:
        rows = [(number, json.loads(line)) for number, line in enumerate(lines, start=1) if line.strip()]
    if not any("evidence" in row for _, row in rows):
        print("No evidence in this results file. Run chatchtg eval with --evidence.", file=sys.stderr)
        return 2
    in_scope = [(number, row) for number, row in rows if "locations" in row.get("context", {})]
    picked = in_scope[:: args.every][: args.limit]
    chosen = {number for number, _ in picked}
    chosen |= {number for number, row in in_scope if args.paths and row["path"].endswith(tuple(args.paths))}

    root = args.dataset_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for number, row in in_scope:
        if number not in chosen:
            continue
        source = (root / row["path"]).resolve()
        if not source.is_relative_to(root):  # the results file is data: stay inside the dataset
            print(f"{number:03d} skipped: {row['path']} leaves the dataset folder", file=sys.stderr)
            continue
        evidence = row.get("evidence") or []
        try:
            image = render_evidence(source.read_bytes(), evidence)
        except OSError as error:  # missing, or not an image
            print(f"{number:03d} skipped: {error}", file=sys.stderr)
            continue
        target = args.out_dir / f"{number:03d}-{source.stem}.png"
        image.save(target)
        print(
            f"{number:03d} {target.name} | expected {row['expected']} | answered {row['extracted_value']} "
            f"| marks {len(evidence)} | {row.get('evidence_note') or 'no note'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
