"""Score the router on stored measurements.

Usage: python -m chatchtg.rooms.report <dataset_dir> <measurements.jsonl> [llm-results.jsonl ...]
Prints, per question kind and split, the accuracy of the usual answer and of the router.
With LLM result files (from ``chatchtg eval``), also compares the two on the same questions.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from chatchtg.rooms.router import decide
from chatchtg.rooms.split import room_questions, split_of, tuned_plans


# The most frequent answer of each kind over the whole benchmark.
USUAL = {
    "bedroom": 1, "sauna": 1, "kitchen": 1, "bathroom": 1, "wc": 0, "balcony": 0, "storeys": 1,
    "label": 1, "apartments": 2,
}  # fmt: skip


def load(dataset_dir: Path, measurements: Path) -> list[dict]:
    """The measured questions, each with ``split``, ``pred``, ``method`` and ``confident``."""
    tuned = tuned_plans(room_questions(dataset_dir))
    rows = []
    failed = 0
    with measurements.open(encoding="utf-8") as lines:
        for line in lines:
            row = json.loads(line)
            if "error" in row["signals"]:
                failed += 1
                continue
            row["split"] = split_of(row["path"], tuned)
            row["pred"], row["method"], row["confident"] = decide(row["kind"], row["signals"])
            rows.append(row)
    if failed:
        # Said, because the accuracy below is then the accuracy on the questions that worked.
        print(f"Left out {failed} questions whose measurement failed.", file=sys.stderr)
    return rows


def _rate(hits: int, total: int) -> str:
    return f"{hits / total:6.1%}" if total else "     -"


def table(rows: list[dict]) -> None:
    by_kind: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_kind[row["kind"]].append(row)
    usual = USUAL
    print(f"{'kind':<11}{'split':<7}{'n':>6}{'usual':>8}{'router':>8}{'sure':>7}{'acc|sure':>9}{'acc|unsure':>11}")
    totals: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for kind in sorted(by_kind):
        for split in ("tuned", "dev", "test"):
            group = [r for r in by_kind[kind] if r["split"] == split]
            if not group:
                continue
            sure = [r for r in group if r["confident"]]
            unsure = [r for r in group if not r["confident"]]
            right = sum(r["pred"] == r["answer"] for r in group)
            constant = sum(usual[kind] == r["answer"] for r in group)
            totals[split][0] += len(group)
            totals[split][1] += constant
            totals[split][2] += right
            print(
                f"{kind:<11}{split:<7}{len(group):>6}{_rate(constant, len(group)):>8}"
                f"{_rate(right, len(group)):>8}{_rate(len(sure), len(group)):>7}"
                f"{_rate(sum(r['pred'] == r['answer'] for r in sure), len(sure)):>9}"
                f"{_rate(sum(r['pred'] == r['answer'] for r in unsure), len(unsure)):>11}"
            )
    for split, (total, constant, right) in totals.items():
        print(f"{'ALL':<11}{split:<7}{total:>6}{_rate(constant, total):>8}{_rate(right, total):>8}")


def compare(rows: list[dict], llm_files: list[Path]) -> None:
    """Router against LLM answers on the questions that both have."""
    llm = {}
    for path in llm_files:
        with path.open(encoding="utf-8") as lines:
            for line in lines:
                row = json.loads(line)
                llm[(row["path"], row["question"])] = row
    by_kind: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    for row in rows:
        other = llm.get((row["path"], row["question"]))
        if other is not None:
            by_kind[row["kind"]].append((row, other))
    print(f"\n{'kind':<11}{'n':>5}{'llm':>8}{'router':>8}{'hybrid':>8}{'sure':>7}{'both ok':>8}{'only llm':>9}{'only rtr':>9}")
    total = Counter()
    for kind in sorted(by_kind):
        pairs = by_kind[kind]
        count = Counter(n=len(pairs))
        for row, other in pairs:
            llm_ok = bool(other["correct"])
            router_ok = row["pred"] == row["answer"]
            count["llm"] += llm_ok
            count["router"] += router_ok
            count["hybrid"] += router_ok if row["confident"] else llm_ok
            count["sure"] += row["confident"]
            count["both"] += llm_ok and router_ok
            count["only_llm"] += llm_ok and not router_ok
            count["only_router"] += router_ok and not llm_ok
        total.update(count)
        _compare_line(kind, count)
    _compare_line("ALL", total)
    cost = sum(row.get("cost_usd") or 0 for row in llm.values())
    print(f"LLM cost of these answers: ${cost:.2f} for {len(llm)} questions")


def _compare_line(kind: str, count: Counter) -> None:
    n = count["n"]
    print(
        f"{kind:<11}{n:>5}{_rate(count['llm'], n):>8}{_rate(count['router'], n):>8}"
        f"{_rate(count['hybrid'], n):>8}{_rate(count['sure'], n):>7}"
        f"{count['both']:>8}{count['only_llm']:>9}{count['only_router']:>9}"
    )


def main(argv: list[str]) -> int:
    rows = load(Path(argv[0]).resolve(), Path(argv[1]))
    table(rows)
    if len(argv) > 2:
        compare(rows, [Path(name) for name in argv[2:]])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
