"""Score the object counter on the benchmark, from the stored detections. No LLM call.

Usage:
    python -m chatchtg.objects.report <questions.jsonl> [--replay <eval-run.jsonl> ...] [--llm <eval-run.jsonl> ...]
        [--version N]

The first table is the exact-count accuracy of every CubiCasa5K object-count question of
the benchmark, per kind and per split of CubiCasa5K. The detector was trained on the train
split and its minimum scores were chosen on the valid split: only the test column measures
plans that neither saw.

``--replay`` takes the rows of an eval run (``chatchtg eval``), replaces the answer of each
question that the counter answers by the counter's answer, and prints the score before
and after, with the split of each replaced plan.

``--llm`` takes any eval result files, keeps every object-count answer of Opus on a plan with
stored detections, and counts per kind and split where the LLM, the counter, both or neither
were right.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from chatchtg.config import Settings
from chatchtg.extractions.models import Extraction
from chatchtg.extractions.store import ExtractionStore
from chatchtg.objects import counter
from chatchtg.objects.counter import PRODUCER, count
from chatchtg.objects.questions import KINDS, classify

SPLITS = ("train", "valid", "test")


def split_of(extraction: Extraction, path: str) -> str:
    """The CubiCasa5K split of the plan at ``path``; ``train`` when the detector was trained on
    the same image under another plan (a few images are in CubiCasa5K twice)."""
    split = extraction.attributes.get("cubicasa_splits", {}).get(path.rsplit("/", 1)[-1])
    split = split or extraction.attributes.get("cubicasa_split", "?")
    return "train" if extraction.attributes.get("in_training_data") else str(split)


def by_source(store: ExtractionStore) -> dict[str, Extraction]:
    """Benchmark path (``data/cubicasa5k/<image>``) -> the stored detections of that plan."""
    found = {}
    for extraction in store.iter(PRODUCER, counter.PRODUCER_VERSION):
        folder = (extraction.source or "").rsplit("/", 1)[0]
        for image in extraction.attributes.get("cubicasa_splits", {}):
            found[f"{folder}/{image}"] = extraction
    return found


def score_benchmark(questions: Path, extractions: dict[str, Extraction]) -> dict:
    """(kind, split) -> questions and exact counts; and the questions without detections."""
    cells: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    missing: Counter[str] = Counter()
    with questions.open(encoding="utf-8") as lines:
        for line in lines:
            row = json.loads(line)
            question = classify(row["question"])
            if question is None:
                continue
            extraction = extractions.get(row["path"])
            if extraction is None:
                missing[row["path"]] += 1
                continue
            finding = count(extraction, question)
            cell = cells[(question.kind, split_of(extraction, row["path"]))]
            cell["n"] += 1
            cell["exact"] += finding.count == row["answer"]
            cell["abs_error"] += abs(finding.count - row["answer"])
    return {"cells": cells, "missing": missing}


def _rate(hits: int, total: int) -> str:
    return f"{hits / total:7.1%}" if total else "      -"


def print_benchmark(result: dict) -> None:
    cells = result["cells"]
    print("Exact counts of the object counter on the benchmark's CubiCasa5K questions")
    print("(English and Finnish questions together; both have the same answer on a plan).")
    header = "".join(f"{split:>16}" for split in SPLITS)
    print(f"{'kind':<13}{header}{'all (inflated)':>16}")
    totals: dict[str, Counter[str]] = defaultdict(Counter)
    for kind in [*KINDS, "ALL"]:
        line = f"{kind:<13}"
        all_cell: Counter[str] = Counter()
        for split in SPLITS:
            cell = totals[split] if kind == "ALL" else cells.get((kind, split), Counter())
            if kind != "ALL":
                totals[split].update(cell)
            all_cell.update(cell)
            line += f"{_rate(cell['exact'], cell['n'])} of {cell['n']:>5}"
        line += f"{_rate(all_cell['exact'], all_cell['n'])} of {all_cell['n']:>5}"
        print(line)
    print("Mean absolute error of the count:")
    for kind in KINDS:
        errors = [
            f"{cells[(kind, s)]['abs_error'] / cells[(kind, s)]['n']:.2f}" if cells.get((kind, s)) else "-"
            for s in SPLITS
        ]
        print(f"{kind:<13}" + "".join(f"{e:>16}" for e in errors))
    print(
        "train: the detector was trained on these plans (and on the few valid and test plans "
        "whose image is also a train plan, counted here), so this column is inflated. "
        "valid: the minimum scores were chosen on these plans. "
        "test: neither; this is the number to quote. "
        "all: mostly train plans, so it is inflated too and is not a result."
    )
    missing = result["missing"]
    if missing:
        print(f"No stored detections for {len(missing)} plans ({sum(missing.values())} questions): "
              "the LLM answers these.")


def replay(run_file: Path, store: ExtractionStore) -> None:
    """Replace the counter's questions in one eval run and print the score before and after."""
    rows = [json.loads(line) for line in run_file.open(encoding="utf-8") if line.strip()]
    before = after = 0
    replaced = []
    for row in rows:
        question = classify(row["question"])
        extraction = None
        if question is not None:
            extraction = store.get(row["document_id"], PRODUCER, counter.PRODUCER_VERSION)
        ok = bool(row.get("correct"))
        before += ok
        if extraction is None:
            after += ok
            continue
        finding = count(extraction, question)
        right = finding.count == row["expected"]
        after += right
        replaced.append((row, question.kind, split_of(extraction, row["path"]), finding.count, ok, right))
    n = len(rows)
    print(f"\n{run_file.name}: {n} questions")
    print(f"  before: {before} of {n} ({before / n:.1%})")
    print(f"  after:  {after} of {n} ({after / n:.1%}), {len(replaced)} answers replaced")
    print(f"  {'plan':<40}{'kind':<13}{'split':<7}{'expected':>9}{'llm':>6}{'counter':>8}")
    for row, kind, split, value, llm_ok, right in replaced:
        llm_value = row.get("extracted_value")
        llm_text = f"{llm_value:g}" if isinstance(llm_value, (int, float)) else "-"
        mark = {(True, True): "", (False, True): "  fixed", (True, False): "  broken", (False, False): ""}
        print(
            f"  {row['path'].rsplit('/', 1)[-1]:<40}{kind:<13}{split:<7}{row['expected']:>9}"
            f"{llm_text:>6}{value:>8}{mark[(llm_ok, right)]}"
        )
    by_split = Counter(split for _, _, split, _, _, _ in replaced)
    print("  replaced plans per split:", dict(by_split))


def compare_llm(files: list[Path], store: ExtractionStore, model_prefix: str = "claude-opus") -> None:
    """Every object-count answer of an LLM in ``files`` against the counter on the same plan."""
    cells: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    seen: set[tuple[str, str, str]] = set()
    plans: set[str] = set()
    for path in files:
        for line in path.open(encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            if not str(row.get("model", "")).startswith(model_prefix) or "document_id" not in row:
                continue
            question = classify(row["question"])
            if question is None:
                continue
            extraction = store.get(row["document_id"], PRODUCER, counter.PRODUCER_VERSION)
            if extraction is None:
                continue
            key = (path.name, row["path"], row["question"])
            if key in seen:
                continue
            seen.add(key)
            plans.add(row["path"])
            llm_ok = bool(row.get("correct"))
            counter_ok = count(extraction, question).count == row["expected"]
            cell = cells[(question.kind, split_of(extraction, row["path"]))]
            cell["n"] += 1
            cell["llm"] += llm_ok
            cell["counter"] += counter_ok
            cell["both"] += llm_ok and counter_ok
            cell["neither"] += not llm_ok and not counter_ok
    print(f"\nLLM ({model_prefix}*) answers against the counter on the same questions: "
          f"{len(seen)} answers on {len(plans)} plans, from {len(files)} files")
    print(f"  {'kind':<13}{'split':<7}{'n':>4}{'llm':>6}{'counter':>9}{'both':>6}{'neither':>9}")
    total: Counter[str] = Counter()
    for (kind, split), cell in sorted(cells.items()):
        total.update(cell)
        print(f"  {kind:<13}{split:<7}{cell['n']:>4}{cell['llm']:>6}{cell['counter']:>9}"
              f"{cell['both']:>6}{cell['neither']:>9}")
    print(f"  {'ALL':<20}{total['n']:>4}{total['llm']:>6}{total['counter']:>9}"
          f"{total['both']:>6}{total['neither']:>9}")


def _option(argv: list[str], name: str) -> list[Path]:
    """The file arguments after ``name``, up to the next option."""
    if name not in argv:
        return []
    rest = argv[argv.index(name) + 1 :]
    stop = next((i for i, a in enumerate(rest) if a.startswith("--")), len(rest))
    return [Path(a) for a in rest[:stop]]


def main(argv: list[str]) -> int:
    from chatchtg.wiring import build_extraction_store

    store = build_extraction_store(Settings.from_env())
    if "--version" in argv:
        # Score another stored version than the one that the app reads.
        counter.PRODUCER_VERSION = int(argv[argv.index("--version") + 1])
    print(f"Detections: {PRODUCER}@{counter.PRODUCER_VERSION}")
    questions = Path(argv[0])
    print_benchmark(score_benchmark(questions, by_source(store)))
    for run_file in _option(argv, "--replay"):
        replay(run_file, store)
    llm_files = _option(argv, "--llm")
    if llm_files:
        compare_llm(llm_files, store)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
