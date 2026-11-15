"""Measure, offline, what a cross-check of measured areas against an LLM would change.

The area pipeline states a confident result in code. Where that result rests on a measured
area and the plan gives a reason for doubt (a scale of 1:200 or coarser, a gap in the room
numbering), a second estimate from an LLM that sees the drawing could catch a gross error.
This script replays that policy without building it: the pipeline's value comes from code,
the LLM's estimate from an ordinary ``baseline`` eval run on the same questions.

    signals   analyse the Swiss plans and write, for every area question, the pipeline's
              value and whether the check would be made (no LLM call)
    select    pick one eligible question per plan, as a question file for ``chatchtg eval``
    replay    join an eval run with the signals and count what the check would change

The paths come from the settings (CHATCHTG_DATA_ROOT, CHATCHTG_STATE_DIR); everything the
script writes is under ``<state dir>/area-check/``.

usage: uv run python scripts/eval_area_check.py signals [--plans N] [--seed 0] [--workers 6]
       uv run python scripts/eval_area_check.py select [--n 100] [--seed 0]
       uv run python scripts/eval_area_check.py replay --results <eval run file> [--threshold 0.10]
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from chatchtg.areas.crosscheck import DISAGREEMENT, compare, eligible
from chatchtg.areas.pipeline import PlanAnalysis
from chatchtg.areas.questions import parse_question
from chatchtg.config import Settings
from chatchtg.context.areas import decide
from chatchtg.evals.scoring import score_value

WRONG = 0.02
"""An area further than this from the reference is wrong (the scoring rule)."""
GROSS = 0.10
"""An area further than this from the reference is grossly wrong."""
THRESHOLDS = (0.05, 0.08, 0.10, 0.15, 0.20)
QUESTION_FIELDS = ("path", "question", "answer", "type", "unit")


def analyse(job: tuple[str, str]) -> tuple[str, dict]:
    """Analyse one plan in a worker process and cache the result."""
    import numpy as np
    from PIL import Image

    from chatchtg.areas import labels
    from chatchtg.areas.pipeline import analyse_plan

    image_path, cache_path = job
    labels.OCR_THREADS = 2  # several plans are read side by side
    try:
        image = np.array(Image.open(image_path).convert("RGB"))
        result = analyse_plan(image).to_json()
    except Exception as error:  # keep the run going; the plan has no answers
        result = {"error": repr(error)}
    Path(cache_path).write_text(json.dumps(result))
    return Path(image_path).stem, result


def signal_rows(questions: list[dict[str, Any]], analysis: PlanAnalysis | None) -> list[dict[str, Any]]:
    """One row for each area question of a plan: the pipeline's value and the signals."""
    rows = []
    for question in questions:
        row = {name: question.get(name) for name in QUESTION_FIELDS}
        row |= {
            "stem": Path(question["path"]).stem, "kind": None, "value": None, "confident": False,
            "basis": None, "doubts": [], "scale": analysis.scale if analysis else None,
            "eligible": False, "error": None,
        }
        query = parse_question(question["question"])
        if analysis is not None and query is not None:
            decision = decide(query, analysis)
            value = decision.answer.value
            row |= {
                "kind": query.kind, "value": value, "confident": decision.confident,
                "basis": decision.basis, "doubts": list(decision.doubts),
                "eligible": decision.confident and eligible(decision.basis, decision.doubts),
            }
            if value is not None and question["answer"]:
                row["error"] = (value - question["answer"]) / question["answer"]
        rows.append(row)
    return rows


def signal_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Questions, wrong and gross answers and plans, for the confident measured answers:
    all of them, the eligible ones and the others."""
    measured = [row for row in rows if row["confident"] and row["basis"] == "measured"]
    groups = [
        ("all confident measured answers", measured),
        ("eligible (1:200 or coarser, or a numbering gap)", [row for row in measured if row["eligible"]]),
        ("not eligible", [row for row in measured if not row["eligible"]]),
    ]
    table = []
    for name, group in groups:
        errors = [abs(row["error"]) for row in group if row["error"] is not None]
        table.append(
            {
                "group": name,
                "questions": len(group),
                "wrong": sum(error > WRONG for error in errors),
                "gross": sum(error > GROSS for error in errors),
                "plans": len({row["stem"] for row in group}),
            }
        )
    return table


def select_questions(rows: list[dict[str, Any]], n: int, seed: int) -> tuple[list[dict[str, Any]], int]:
    """One eligible question per plan, picked at random, for ``n`` plans picked at random.

    Returns the question rows, in the random order of the pick, and the number of plans
    that have an eligible question. A question that is known to be wrong is not preferred.
    """
    by_plan: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["eligible"]:
            by_plan[row["stem"]].append(row)
    rng = random.Random(seed)
    picked = [rng.choice(by_plan[stem]) for stem in sorted(by_plan)]
    rng.shuffle(picked)
    return [{name: row[name] for name in QUESTION_FIELDS} for row in picked[:n]], len(by_plan)


def replay_rows(
    signals: list[dict[str, Any]], run: list[dict[str, Any]], threshold: float = DISAGREEMENT
) -> list[dict[str, Any]]:
    """The check for each row of an eval run that has a signal row with a value."""
    by_question = {(row["path"], row["question"]): row for row in signals}
    rows = []
    for answer in run:
        signal = by_question.get((answer["path"], answer["question"]))
        if signal is None or signal["value"] is None:
            continue
        check = compare(signal["value"], answer.get("extracted_value"), threshold)
        expected = answer["expected"]
        llm_value = answer.get("extracted_value")
        rows.append(
            {
                "path": answer["path"], "question": answer["question"], "expected": expected,
                "kind": signal["kind"], "scale": signal["scale"], "doubts": signal["doubts"],
                "tool_value": signal["value"], "llm_value": llm_value,
                "difference": check.difference, "verdict": check.verdict, "chosen": check.chosen,
                "tool_error": (signal["value"] - expected) / expected,
                "llm_error": (llm_value - expected) / expected if llm_value is not None else None,
                "tool_correct": bool(score_value("area", expected, signal["value"])),
                "llm_correct": bool(score_value("area", expected, llm_value)),
                "check_correct": bool(score_value("area", expected, check.chosen)),
                "cost_usd": answer.get("cost_usd"),
            }
        )
    return rows


def replay_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    flagged = [row for row in rows if row["verdict"] == "disagree"]
    return {
        "questions": len(rows),
        "tool_correct": sum(row["tool_correct"] for row in rows),
        "llm_correct": sum(row["llm_correct"] for row in rows),
        "check_correct": sum(row["check_correct"] for row in rows),
        "no_estimate": sum(row["verdict"] == "no-estimate" for row in rows),
        "flagged": len(flagged),
        "tool_wrong_among_flagged": sum(not row["tool_correct"] for row in flagged),
        "became_right": sum(row["check_correct"] and not row["tool_correct"] for row in flagged),
        "became_wrong": sum(row["tool_correct"] and not row["check_correct"] for row in flagged),
    }


def error_summary(errors: list[float]) -> str:
    """The size of a list of relative errors: quantiles and the shares above 2%, 5%, 10%."""
    sizes = sorted(abs(error) for error in errors)
    if not sizes:
        return "no values"

    def quantile(share: float) -> float:
        return sizes[min(len(sizes) - 1, int(share * len(sizes)))]

    above = ", ".join(f">{limit:.0%}: {sum(size > limit for size in sizes)}" for limit in (0.02, 0.05, 0.10, 0.20))
    return (
        f"n {len(sizes)}, median {statistics.median(sizes):.1%}, 75th {quantile(0.75):.1%}, "
        f"90th {quantile(0.90):.1%}, max {sizes[-1]:.1%}; {above}"
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    # Not splitlines(): it also splits on separators that a JSON string may contain.
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def signals(args: argparse.Namespace, settings: Settings, work_dir: Path) -> None:
    dataset = settings.data_root / "floorplans"
    questions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with (dataset / "questions.jsonl").open(encoding="utf-8") as lines:
        for line in lines:
            row = json.loads(line)
            if row["type"] == "area":
                questions[Path(row["path"]).stem].append(row)
    stems = sorted(questions)
    if args.plans is not None:
        random.Random(args.seed).shuffle(stems)
        stems = sorted(stems[: args.plans])

    cache = work_dir / "analyses"
    cache.mkdir(parents=True, exist_ok=True)
    analyses: dict[str, dict] = {}
    todo = []
    for stem in stems:
        cached = cache / f"{stem}.json"
        if cached.is_file():
            analyses[stem] = json.loads(cached.read_text())
        else:
            image = dataset / questions[stem][0]["path"]
            todo.append((str(image), str(cached)))
    if todo:
        print(f"analysing {len(todo)} plans ({len(stems) - len(todo)} from the cache)", flush=True)
        with ProcessPoolExecutor(args.workers) as pool:
            for done, (stem, result) in enumerate(pool.map(analyse, todo), start=1):
                analyses[stem] = result
                if done % 50 == 0:
                    print(f"  {done} of {len(todo)}", flush=True)

    rows = []
    failed = 0
    for stem in stems:
        result = analyses[stem]
        failed += "error" in result
        analysis = None if "error" in result else PlanAnalysis.from_json(result)
        rows.extend(signal_rows(questions[stem], analysis))
    write_jsonl(work_dir / "signals.jsonl", rows)

    confident = sum(row["confident"] for row in rows)
    print(f"plans: {len(stems)} ({failed} that the pipeline could not analyse), area questions: {len(rows)}")
    print(f"answered by the pipeline with confidence: {confident}")
    print(f"{'group':50s} {'questions':>9s} {'wrong (>2%)':>12s} {'gross (>10%)':>13s} {'plans':>6s}")
    for line in signal_table(rows):
        print(f"{line['group']:50s} {line['questions']:9d} {line['wrong']:12d} {line['gross']:13d} {line['plans']:6d}")
    table = {line["group"]: line for line in signal_table(rows)}
    chosen = next(line for name, line in table.items() if name.startswith("eligible"))
    if chosen["questions"]:
        print(
            f"eligible: {chosen['questions'] / len(rows):.1%} of the area questions; "
            f"gross among the eligible: {chosen['gross'] / chosen['questions']:.1%}"
        )
    print(f"signals: {work_dir / 'signals.jsonl'}")


def select(args: argparse.Namespace, settings: Settings, work_dir: Path) -> None:
    rows, plans = select_questions(read_jsonl(work_dir / "signals.jsonl"), args.n, args.seed)
    write_jsonl(work_dir / "questions.jsonl", rows)
    print(f"plans with an eligible question: {plans}; questions written: {len(rows)}")
    print(f"questions: {work_dir / 'questions.jsonl'}")


def replay(args: argparse.Namespace, settings: Settings, work_dir: Path) -> None:
    signal_list = read_jsonl(work_dir / "signals.jsonl")
    run = read_jsonl(args.results)
    rows = replay_rows(signal_list, run, args.threshold)
    write_jsonl(work_dir / "replay.jsonl", rows)

    counts = replay_counts(rows)
    n = counts["questions"]
    costs = [row["cost_usd"] for row in rows if row["cost_usd"] is not None]
    print(f"rows of the run: {len(run)}, joined with a pipeline value: {n}")
    print(f"threshold: {args.threshold:.0%}")
    print(f"correct, pipeline alone: {counts['tool_correct']} of {n}")
    print(f"correct, LLM alone:      {counts['llm_correct']} of {n}")
    print(f"correct, cross-check:    {counts['check_correct']} of {n}")
    print(f"LLM gave no number: {counts['no_estimate']}")
    print(f"flagged (disagreement): {counts['flagged']}")
    print(f"pipeline wrong among flagged: {counts['tool_wrong_among_flagged']}")
    print(f"became right: {counts['became_right']}, became wrong: {counts['became_wrong']}")
    if costs:
        print(
            f"cost of one check (one LLM answer): mean ${statistics.mean(costs):.4f}, "
            f"median ${statistics.median(costs):.4f}, max ${max(costs):.4f}, total ${sum(costs):.2f}"
        )
    print("error of the LLM against the reference:      " + error_summary([r["llm_error"] for r in rows if r["llm_error"] is not None]))
    print("error of the pipeline against the reference: " + error_summary([r["tool_error"] for r in rows]))
    print("sensitivity (the gate uses 10% only):")
    print(f"{'threshold':>9s} {'flagged':>8s} {'pipeline wrong':>15s} {'became right':>13s} {'became wrong':>13s} {'correct':>8s}")
    for threshold in THRESHOLDS:
        c = replay_counts(replay_rows(signal_list, run, threshold))
        print(
            f"{threshold:9.0%} {c['flagged']:8d} {c['tool_wrong_among_flagged']:15d} "
            f"{c['became_right']:13d} {c['became_wrong']:13d} {c['check_correct']:8d}"
        )
    print(f"per question: {work_dir / 'replay.jsonl'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    signals_parser = commands.add_parser("signals", help="the pipeline's value and signals for every area question")
    signals_parser.add_argument("--plans", type=int, help="this many plans, picked with the seed (default: all)")
    signals_parser.add_argument("--seed", type=int, default=0)
    signals_parser.add_argument("--workers", type=int, default=6)
    signals_parser.set_defaults(run=signals)
    select_parser = commands.add_parser("select", help="one eligible question per plan, as a question file")
    select_parser.add_argument("--n", type=int, default=100)
    select_parser.add_argument("--seed", type=int, default=0)
    select_parser.set_defaults(run=select)
    replay_parser = commands.add_parser("replay", help="what the check would change in an eval run")
    replay_parser.add_argument("--results", type=Path, required=True, help="the result file of a baseline eval run")
    replay_parser.add_argument("--threshold", type=float, default=DISAGREEMENT)
    replay_parser.set_defaults(run=replay)
    args = parser.parse_args()
    settings = Settings.from_env()
    args.run(args, settings, settings.state_dir / "area-check")


if __name__ == "__main__":
    main()
