"""Score the area pipeline end to end on benchmark questions, with no LLM call.

For a sample of Swiss plans the pipeline analyses the PNG (OCR, membership, measurement)
and answers every area question of those plans in code. An answer counts as correct when
it is within 2% of the reference. The DXF truth (var/room-truth, built by
scripts/build_room_truth.py) is used only to say which stage caused an error.

usage: uv run python scripts/eval_areas.py [--plans 100] [--seed 0] [--workers 6] [--refresh]
       uv run python scripts/eval_areas.py --stems FILE --refresh   # exactly these plans

The plans are read from CHATCHTG_DATA_ROOT (default ``data``). The cache in var/area-cache
is keyed by the plan name only: pass --refresh after a change to the pipeline.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from chatchtg.areas import labels
from chatchtg.areas.pipeline import PlanAnalysis, analyse_plan
from chatchtg.areas.questions import answer_query, parse_question
from chatchtg.config import Settings

DATA = Settings.from_env().data_root / "floorplans"
PLANS = DATA / "data" / "swiss_dwellings"
TRUTH = Path("var/room-truth")
CACHE = Path("var/area-cache")
TOLERANCE = 0.02


def analyse(stem: str) -> dict:
    labels.OCR_THREADS = 2  # several plans are read side by side
    image = np.array(Image.open(PLANS / f"{stem}.png").convert("RGB"))
    return analyse_plan(image).to_json()


def cached(stem: str, refresh: bool) -> dict | None:
    path = CACHE / f"{stem}.json"
    if path.is_file() and not refresh:
        return json.loads(path.read_text())
    return None


def work(stem: str) -> tuple[str, dict]:
    try:
        result = analyse(stem)
    except Exception as error:  # keep the run going; the plan counts as unanswered
        result = {"error": repr(error)}
    (CACHE / f"{stem}.json").write_text(json.dumps(result))
    return stem, result


def within(value: float | None, reference: float) -> bool:
    return value is not None and abs(value - reference) <= TOLERANCE * abs(reference)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plans", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--refresh", action="store_true", help="analyse again, ignore the cache")
    parser.add_argument("--exclude", type=Path, help="file with one plan stem per line to leave out")
    parser.add_argument(
        "--stems", type=Path,
        help="file with one plan stem per line: score exactly these plans (--plans, --seed and --exclude are ignored)",
    )
    parser.add_argument("--results", type=Path, help="an eval-run file: score its area questions, next to its own answers")
    parser.add_argument("--out", type=Path, default=Path("var/eval-areas.jsonl"))
    args = parser.parse_args()

    questions = defaultdict(list)
    with (DATA / "questions.jsonl").open(encoding="utf-8") as lines:
        for line in lines:
            row = json.loads(line)
            if row["type"] == "area":
                questions[Path(row["path"]).stem].append(row)
    baseline: dict[tuple[str, str], bool] = {}
    if args.results:
        questions.clear()
        for line in args.results.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["type"] == "area":
                stem = Path(row["path"]).stem
                questions[stem].append({"question": row["question"], "answer": row["expected"]})
                baseline[(stem, row["question"])] = bool(row["correct"])
        args.plans = len(questions)
    stems = sorted(questions)
    if args.exclude:
        left_out = set(args.exclude.read_text().split())
        stems = [s for s in stems if s not in left_out]
    random.Random(args.seed).shuffle(stems)
    stems = stems[: args.plans]
    if args.stems:
        stems = [s for s in args.stems.read_text().split() if s in questions]

    CACHE.mkdir(parents=True, exist_ok=True)
    analyses: dict[str, dict] = {}
    todo = []
    for stem in stems:
        hit = cached(stem, args.refresh)
        if hit is None:
            todo.append(stem)
        else:
            analyses[stem] = hit
    if todo:
        with ProcessPoolExecutor(args.workers) as pool:
            for stem, result in pool.map(work, todo):
                analyses[stem] = result

    counts: dict[tuple, Counter] = defaultdict(Counter)
    causes: Counter = Counter()
    rows = []
    for stem in stems:
        result = analyses[stem]
        truth = json.loads((TRUTH / f"{stem}.json").read_text()) if (TRUTH / f"{stem}.json").is_file() else None
        printed = bool(truth) and any(r["printed_area"] is not None for r in truth["rooms"])
        analysis = None if "error" in result else PlanAnalysis.from_json(result)
        for question in questions[stem]:
            query = parse_question(question["question"])
            value, complete, used = None, False, []
            if analysis is not None and query is not None:
                answer = answer_query(query, analysis.rows())
                value = answer.value
                complete = value is not None and not analysis.may_be_incomplete(query.kind)
                used = [r.number for r in answer.rooms]
            kind = query.kind if query else "unparsed"
            correct = within(value, question["answer"])
            scale = analysis.scale if analysis else (truth or {}).get("scale")
            for key in (("all",), (kind,), ("printed" if printed else "unprinted",), (kind, "printed" if printed else "unprinted"), (f"1:{scale}",)):
                c = counts[key]
                c["n"] += 1
                c["correct"] += correct
                c["confident"] += complete
                c["confident_correct"] += complete and correct
                c["baseline"] += baseline.get((stem, question["question"]), False)
            if not correct:
                causes[cause(analysis, truth, query, used, value)] += 1
            rows.append({"stem": stem, "question": question["question"], "expected": question["answer"],
                         "value": value, "correct": correct, "confident": complete, "kind": kind, "printed": printed})
    args.out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))

    print(f"plans: {len(stems)}, area questions: {counts[('all',)]['n']}")
    print(f"{'group':28s} {'n':>6s} {'correct':>8s} {'confident':>10s} {'correct when confident':>24s}")
    for key in sorted(counts, key=lambda k: (len(k), k)):
        c = counts[key]
        sure = f"{c['confident_correct'] / c['confident']:.1%}" if c["confident"] else "-"
        other = f"   run file: {c['baseline']}/{c['n']}, pipeline: {c['correct']}/{c['n']}" if baseline else ""
        print(f"{' / '.join(key):28s} {c['n']:6d} {c['correct'] / c['n']:8.1%} {c['confident'] / c['n']:10.1%} {sure:>24s}{other}")
    print("causes of wrong answers:")
    for name, n in causes.most_common():
        print(f"  {name:40s} {n}")


def cause(analysis, truth, query, used, value) -> str:
    """The first stage that explains a wrong answer, by comparison with the DXF truth."""
    if analysis is None:
        return "pipeline error"
    if query is None:
        return "question not parsed"
    if truth is None:
        return "no truth"
    true_rooms = {r["number"]: r for r in truth["rooms"]}
    found = {r.number: r for r in analysis.rooms}
    if not analysis.scale_read or analysis.scale != truth["scale"]:
        return "scale not read"
    # the rooms that should have been added up
    def outdoor(room) -> bool:
        got = found.get(room["number"])
        return got.outdoor if got else False
    if query.kind == "room":
        wanted = [query.number] if query.number in true_rooms else []
    elif query.kind == "apartment":
        wanted = [n for n, r in true_rooms.items() if r["apartment"] == query.apartment and not outdoor(r)]
    elif query.kind == "floor":
        wanted = [n for n, r in true_rooms.items() if not outdoor(r)]
    else:
        wanted = used
    missing = [n for n in wanted if n not in found]
    if missing:
        return "room label not read"
    extra = [n for n in used if n not in true_rooms]
    if extra:
        return "room number misread"
    if query.kind in ("apartment", "floor") and set(used) != set(wanted):
        return "wrong apartment or outdoor flag" if query.kind == "apartment" else "wrong outdoor flag"
    if value is None:
        return "room without area"
    bad = [n for n in used if n in true_rooms and found[n].area is not None
           and abs(found[n].area - true_rooms[n]["area"]) > TOLERANCE * true_rooms[n]["area"]]
    if bad:
        methods = {found[n].method for n in bad}
        return "area off: " + "+".join(sorted(m or "none" for m in methods))
    return "other (rounding, label match)"


if __name__ == "__main__":
    main()
