"""Score every benchmark question that a code path answers, on one table. No LLM call.

    uv run python -m chatchtg.evals.code_paths data/floorplans \\
        --area-cache var/area-cache --rooms var/rooms.jsonl \\
        --area-tuned docs/benchmarks/stems/tune-100.txt docs/benchmarks/stems/tune-200.txt var/label-dev-stems.txt \\
        --llm <eval-run.jsonl> ... --out var/code-paths.jsonl

Four code paths answer questions without an LLM. Each claims a set of questions by their
wording; this module checks that no question is claimed twice, takes each path's stored answer,
and scores it with the eval rule (``scoring.score_value``):

- ``areas``: the area pipeline (``areas/``). Its answers are computed from the plan analyses
  that ``scripts/eval_areas.py`` caches in ``var/area-cache``, as the areas composer states them.
- ``rooms``: the room counter (``rooms/``). Its measurements are read from the output of
  ``python -m chatchtg.rooms.bench``; ``rooms.router.decide`` turns them into the count.
- ``objects-cubicasa`` and ``objects-swiss``: the object counter (``objects/``), counted from
  the detections in the extraction store (``var/metadata``), as ``objects.report`` does.

Each path was fitted on some plans. Every answered question carries the split of its plan for
its path, and ``held_out`` is True only on plans that the path was not fitted on:

- objects: the detector's ``test`` split (``train`` was trained on, ``valid`` chose the
  minimum scores);
- rooms: the ``test`` split of ``rooms/split.py`` (``tuned`` tuned the lexicon, ``dev`` was
  left for further tuning);
- areas: every plan that is not in the given tuning lists (the measurer's 40 plans and the
  label reader's development sample).

The report gives, per kind of question, the accuracy on the held-out plans (the number to
quote) and on all plans (inflated), the code-only score of the whole benchmark with an interval
from a bootstrap over plans, and, with ``--llm``, the Opus answers in existing eval-run files
per kind, the estimated plain-LLM score and the score of the routing in ``ROUTING``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from chatchtg.areas.questions import parse_question
from chatchtg.evals.scoring import score_value
from chatchtg.objects import questions as object_questions
from chatchtg.objects.counter import PROFILES
from chatchtg.rooms import questions as room_questions

DATASETS = {"cubicasa5k": "finnish", "swiss_dwellings": "swiss"}
"""The folder of a plan in the benchmark's paths -> the name used in kinds."""

# Every kind of question, in the order of the report, with its name in the tables.
KINDS: dict[str, str] = {
    "finnish:window": "Finnish: windows",
    "finnish:door": "Finnish: doors",
    "finnish:toilet": "Finnish: toilets",
    "finnish:sink": "Finnish: sinks and wash basins",
    "finnish:bath_shower": "Finnish: showers and bathtubs",
    "finnish:bedroom": "Finnish: bedrooms",
    "finnish:kitchen": "Finnish: kitchens",
    "finnish:bathroom": "Finnish: bathrooms and washrooms",
    "finnish:wc": "Finnish: separate WC rooms",
    "finnish:balcony": "Finnish: balconies",
    "finnish:sauna": "Finnish: saunas",
    "finnish:storeys": "Finnish: storeys",
    "swiss:label": "Swiss: rooms with a given label",
    "swiss:apartments": "Swiss: apartments",
    "swiss:window": "Swiss: windows",
    "swiss:door": "Swiss: doors",
    "swiss:toilet": "Swiss: toilets",
    "swiss:washbasin": "Swiss: washbasins",
    "swiss:bathtub": "Swiss: bathtubs",
    "swiss:shower": "Swiss: showers",
    "swiss:area-room": "Swiss: area of one numbered room",
    "swiss:area-label": "Swiss: total area of rooms with a label",
    "swiss:area-apartment": "Swiss: area of one apartment",
    "swiss:area-floor": "Swiss: net floor area of the floor",
}

HELD_OUT_SPLITS = {"areas": "held-out", "rooms": "test", "objects-cubicasa": "test", "objects-swiss": "test"}
"""The split of each path whose plans the path was not fitted on."""


@dataclass(frozen=True)
class Claim:
    path: str
    """The code path: ``areas``, ``rooms``, ``objects-cubicasa`` or ``objects-swiss``."""
    kind: str
    """One of KINDS."""


@dataclass(frozen=True)
class Answer:
    """What a code path answered to one question, read from its stored results."""

    value: float | None
    """None when the path ran and gave no number (scored as wrong)."""
    split: str
    """The split of the plan for this path (see HELD_OUT_SPLITS)."""
    confident: bool | None = None
    """The path's own flag, where it has one (areas, rooms)."""


@dataclass
class Scored:
    plan: str
    question: str
    path: str
    kind: str
    expected: Any
    value: float | None
    correct: bool
    split: str
    held_out: bool
    confident: bool | None


def dataset_of(plan: str) -> str:
    """``data/cubicasa5k/x.png`` -> ``finnish``."""
    parts = Path(plan).parts
    folder = parts[1] if len(parts) > 2 else parts[0]
    return DATASETS.get(folder, folder)


def claims(row: Mapping[str, Any]) -> list[Claim]:
    """Every code path that claims the benchmark question ``row`` (``path``, ``question``,
    ``type``). The paths are meant to be disjoint; the caller checks that at most one claims."""
    message = row["question"]
    dataset = dataset_of(row["path"])
    found = []
    query = parse_question(message)
    if query is not None and dataset == "swiss":
        found.append(Claim("areas", f"swiss:area-{query.kind}"))
    room = room_questions.classify(message) if row.get("type", "count") == "count" else None
    if room is not None:
        profile = "swiss" if room.kind in room_questions.SWISS_KINDS else "finnish"
        if profile == dataset:
            found.append(Claim("rooms", f"{dataset}:{room.kind}"))
    thing = object_questions.classify(message)
    if thing is not None:
        for profile in PROFILES:
            if row["path"].startswith(profile.source_folder + "/") and thing.kind in profile.classes:
                found.append(Claim(f"objects-{profile.name}", f"{dataset}:{thing.kind}"))
                break
    return found


class DoubleClaimError(ValueError):
    """A question is claimed by more than one code path."""


def merge(
    questions: Iterable[Mapping[str, Any]], answers: Mapping[str, Mapping[tuple[str, str], Answer]]
) -> tuple[list[Scored], list[Mapping[str, Any]], list[tuple[Mapping[str, Any], Claim]]]:
    """Score the answer of the path that claims each question.

    ``answers`` is path -> (plan, question) -> Answer. Returns the scored questions, the
    questions that no path claims, and the claimed questions that their path has no stored
    answer for (not run). Raises DoubleClaimError when two paths claim one question.
    """
    scored: list[Scored] = []
    unclaimed: list[Mapping[str, Any]] = []
    not_run: list[tuple[Mapping[str, Any], Claim]] = []
    for row in questions:
        found = claims(row)
        if len(found) > 1:
            raise DoubleClaimError(f"{row['question']!r} on {row['path']} is claimed by {found}")
        if not found:
            unclaimed.append(row)
            continue
        claim = found[0]
        answer = answers.get(claim.path, {}).get((row["path"], row["question"]))
        if answer is None:
            not_run.append((row, claim))
            continue
        correct = score_value(row["type"], row["answer"], answer.value)
        scored.append(
            Scored(
                plan=row["path"], question=row["question"], path=claim.path, kind=claim.kind,
                expected=row["answer"], value=answer.value, correct=bool(correct), split=answer.split,
                held_out=answer.split == HELD_OUT_SPLITS[claim.path], confident=answer.confident,
            )
        )
    return scored, unclaimed, not_run


# ---- Summaries ----------------------------------------------------------------------------


@dataclass
class KindSummary:
    kind: str
    benchmark: int
    """Questions of this kind in the benchmark."""
    answered: int
    correct: int
    held_out: int
    held_out_correct: int
    held_out_plans: int
    plans: int


def summarise(scored: Iterable[Scored], benchmark: Mapping[str, int]) -> dict[str, KindSummary]:
    """Per kind: the counts on all plans and on the held-out plans."""
    out = {k: KindSummary(k, n, 0, 0, 0, 0, 0, 0) for k, n in benchmark.items()}
    plans: dict[str, set[str]] = defaultdict(set)
    held_plans: dict[str, set[str]] = defaultdict(set)
    for row in scored:
        s = out.setdefault(row.kind, KindSummary(row.kind, 0, 0, 0, 0, 0, 0, 0))
        s.answered += 1
        s.correct += row.correct
        plans[row.kind].add(row.plan)
        if row.held_out:
            s.held_out += 1
            s.held_out_correct += row.correct
            held_plans[row.kind].add(row.plan)
    for kind, s in out.items():
        s.plans = len(plans[kind])
        s.held_out_plans = len(held_plans[kind])
    return out


def weighted_score(accuracy: Mapping[str, float], weights: Mapping[str, int]) -> float:
    """The question-weighted mean of per-kind accuracies over the kinds in ``weights``."""
    total = sum(weights.values())
    return sum(weights[k] * accuracy[k] for k in weights) / total


def bootstrap(
    scored: Iterable[Scored], weights: Mapping[str, int], draws: int = 2000, seed: int = 0,
    held_out_only: bool = True,
) -> tuple[float, float, float]:
    """The question-weighted score and its 95% interval, resampling plans, not questions.

    Each draw resamples the plans of each dataset with replacement (a plan keeps all its
    questions, of every kind and every path), recomputes each kind's accuracy on the held-out
    questions of the drawn plans and weights it by ``weights``. Returns (point, low, high).
    """
    import numpy as np

    kinds = list(weights)
    index = {k: i for i, k in enumerate(kinds)}
    cells: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for row in scored:
        if (held_out_only and not row.held_out) or row.kind not in index:
            continue
        cell = cells[row.plan][row.kind]
        cell[0] += 1
        cell[1] += row.correct
    plans = sorted(cells)
    n = np.zeros((len(plans), len(kinds)))
    c = np.zeros((len(plans), len(kinds)))
    for p, plan in enumerate(plans):
        for kind, (count, correct) in cells[plan].items():
            n[p, index[kind]] = count
            c[p, index[kind]] = correct
    w = np.array([weights[k] for k in kinds], dtype=float)
    covered = n.sum(axis=0) > 0
    if not covered.all():
        missing = [k for k, ok in zip(kinds, covered) if not ok]
        raise ValueError(f"No held-out question of these kinds: {missing}")

    def score(counts: np.ndarray, hits: np.ndarray) -> np.ndarray:
        # A draw without a held-out question of some kind weights the other kinds up.
        accuracy = hits / np.where(counts > 0, counts, np.nan)
        present = np.where(np.isnan(accuracy), 0.0, w)
        return np.nansum(accuracy * w, axis=-1) / present.sum(axis=-1)

    point = float(score(n.sum(axis=0), c.sum(axis=0)))
    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    for p, plan in enumerate(plans):
        groups[dataset_of(plan)].append(p)
    draws_w = np.zeros((draws, len(plans)))
    for members in groups.values():
        picks = rng.multinomial(len(members), [1 / len(members)] * len(members), size=draws)
        draws_w[:, members] = picks
    values = score(draws_w @ n, draws_w @ c)
    low, high = np.percentile(values, [2.5, 97.5])
    return point, float(low), float(high)


def wilson(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """The 95% Wilson interval of ``hits`` of ``n``."""
    if n == 0:
        return 0.0, 1.0
    p = hits / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


# ---- LLM answers in existing eval runs -----------------------------------------------------


@dataclass
class LlmTally:
    answers: int = 0
    correct: int = 0
    cost: float = 0.0
    questions: set = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.questions is None:
            self.questions = set()


_VERSION = re.compile(r"@(\d+)$")


def prompt_of(row: Mapping[str, Any], kind: str) -> str | None:
    """How the LLM was asked: ``plain`` (the baseline context), ``conventions`` (the room
    conventions of a Finnish room count) or None (the LLM only stated a result that code
    computed, so the row measures no LLM)."""
    composer = str(row.get("composer", ""))
    if composer.startswith("baseline@"):
        return "plain"
    if composer.startswith("areas@"):
        # The areas composer changes area questions only; it sends every other one as the baseline does.
        return None if kind.startswith("swiss:area-") else "plain"
    if composer.startswith("room-conventions@"):
        finnish_room = kind.startswith("finnish:") and kind.split(":")[1] in room_questions.FINNISH_KINDS
        return "conventions" if finnish_room else "plain"
    return None


def tally_llm(rows: Iterable[Mapping[str, Any]], model_prefix: str = "claude-opus") -> dict[tuple[str, str], LlmTally]:
    """(kind, prompt) -> the LLM's answers. For ``conventions`` only the latest version of the
    prompt per question is kept (the earlier versions were replaced for a reason)."""
    tallies: dict[tuple[str, str], LlmTally] = defaultdict(LlmTally)
    latest: dict[tuple[str, str], tuple[int, list[Mapping[str, Any]]]] = {}
    for row in rows:
        if not str(row.get("model", "")).startswith(model_prefix) or row.get("correct") is None:
            continue
        found = claims({"path": row["path"], "question": row["question"], "type": row["type"]})
        if len(found) != 1:
            continue
        kind = found[0].kind
        prompt = prompt_of(row, kind)
        if prompt is None:
            continue
        if prompt == "conventions":
            match = _VERSION.search(str(row["composer"]))
            version = int(match.group(1)) if match else 0
            key = (row["path"], row["question"])
            seen = latest.get(key)
            if seen is None or version > seen[0]:
                latest[key] = (version, [row])
            elif version == seen[0]:
                seen[1].append(row)
            continue
        _add(tallies[(kind, prompt)], row)
    for _, kept in latest.values():
        for row in kept:
            kind = claims(row)[0].kind
            _add(tallies[(kind, "conventions")], row)
    return tallies


def _add(tally: LlmTally, row: Mapping[str, Any]) -> None:
    tally.answers += 1
    tally.correct += bool(row["correct"])
    tally.cost += float(row.get("cost_usd") or 0.0)
    tally.questions.add((row["path"], row["question"]))


def read_runs(files: Iterable[Path]) -> tuple[list[dict], list[Path]]:
    """The rows of the eval-run files; a file whose bytes equal an earlier one is read once."""
    rows, used, seen = [], [], set()
    for path in files:
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        used.append(path)
        for line in data.decode("utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows, used


# ---- Routing -----------------------------------------------------------------------------

# Kind -> (who answers, evidence class). ``code-confident``: code when the path says it is
# confident, the LLM otherwise (CHATCHTG_ROOM_COUNTER=confident for apartments, the areas
# composer for area questions). The reasons are in docs/benchmarks/2026-11-15-whole-benchmark.md.
SAME = "measured on the same questions"
HANDFUL = "LLM side a handful of questions"
NONE = "LLM not measured; chosen by reasoning"
ROUTING: dict[str, tuple[str, str]] = {
    "finnish:window": ("code", HANDFUL),
    "finnish:door": ("code", HANDFUL),
    "finnish:toilet": ("code", HANDFUL),
    "finnish:sink": ("code", HANDFUL),
    "finnish:bath_shower": ("code", HANDFUL),
    "finnish:bedroom": ("llm", SAME),
    "finnish:kitchen": ("llm", SAME),
    "finnish:bathroom": ("llm", SAME),
    "finnish:wc": ("llm", SAME),
    "finnish:balcony": ("llm", SAME),
    "finnish:sauna": ("llm", SAME),
    "finnish:storeys": ("llm", SAME),
    "swiss:label": ("llm", SAME),
    "swiss:apartments": ("code-confident", SAME),
    "swiss:window": ("code", HANDFUL),
    "swiss:door": ("code", NONE),
    "swiss:toilet": ("code", HANDFUL),
    "swiss:washbasin": ("code", HANDFUL),
    "swiss:bathtub": ("code", HANDFUL),
    "swiss:shower": ("code", NONE),
    "swiss:area-room": ("code-confident", HANDFUL),
    "swiss:area-label": ("code-confident", HANDFUL),
    "swiss:area-apartment": ("code-confident", HANDFUL),
    "swiss:area-floor": ("code-confident", HANDFUL),
}


# ---- Loaders of the stored answers (files) -----------------------------------------------


def load_area_answers(
    questions: Iterable[Mapping[str, Any]], cache_dir: Path, tuned: set[str]
) -> dict[tuple[str, str], Answer]:
    """The area pipeline's answer to each area question whose plan is in ``cache_dir`` (the
    analyses that ``scripts/eval_areas.py`` stores, one JSON file per plan).

    The answer is what the app gives: ``context.areas.decide`` says whether the pipeline is
    confident (the app then states the answer in code), and the value is read from the
    statement text as the eval rule reads it (two decimals). A plan whose analysis failed
    answers None. Plans without a cached analysis are left out (not run).
    """
    from chatchtg.areas.pipeline import PlanAnalysis
    from chatchtg.context.areas import decide, statement_text
    from chatchtg.evals.scoring import extract_number

    analyses: dict[str, PlanAnalysis | None] = {}
    found = {}
    for row in questions:
        query = parse_question(row["question"])
        if query is None or dataset_of(row["path"]) != "swiss":
            continue
        stem = Path(row["path"]).stem
        if stem not in analyses:
            cached = cache_dir / f"{stem}.json"
            if not cached.is_file():
                continue
            data = json.loads(cached.read_text())
            analyses[stem] = None if "error" in data else PlanAnalysis.from_json(data)
        analysis = analyses[stem]
        split = "tuned" if stem in tuned else "held-out"
        if analysis is None:
            found[(row["path"], row["question"])] = Answer(None, split, False)
            continue
        decision = decide(query, analysis)
        value = None
        if decision.answer.value is not None:
            value = extract_number(statement_text(query, analysis, decision))
        found[(row["path"], row["question"])] = Answer(value, split, decision.confident)
    return found


def load_room_answers(dataset_dir: Path, measurements: Path) -> dict[tuple[str, str], Answer]:
    """The output of ``rooms.bench``, decided as ``rooms.report`` does."""
    from chatchtg.rooms.report import load

    return {
        (row["path"], row["question"]): Answer(float(row["pred"]), row["split"], bool(row["confident"]))
        for row in load(dataset_dir, measurements)
    }


def load_object_answers(questions: Iterable[Mapping[str, Any]], store) -> dict[str, dict[tuple[str, str], Answer]]:
    """The object counter's count of each object question, from the stored detections."""
    from chatchtg.objects.counter import count
    from chatchtg.objects.report import by_source, split_of

    out: dict[str, dict[tuple[str, str], Answer]] = {}
    for profile in PROFILES:
        extractions = by_source(store, profile)
        answers = out[f"objects-{profile.name}"] = {}
        for row in questions:
            thing = object_questions.classify(row["question"])
            if thing is None or thing.kind not in profile.classes:
                continue
            extraction = extractions.get(row["path"])
            if extraction is None:
                continue
            finding = count(extraction, thing)
            answers[(row["path"], row["question"])] = Answer(float(finding.count), split_of(extraction, row["path"]))
    return out


# ---- Report -------------------------------------------------------------------------------


def _pct(hits: int, n: int) -> str:
    return f"{hits / n:.1%}" if n else "-"


def report(
    scored: list[Scored], benchmark: Counter, tallies: dict[tuple[str, str], LlmTally] | None
) -> dict[str, Any]:
    summary = summarise(scored, benchmark)
    print("\n## Code paths per kind\n")
    print("| Kind | Benchmark | Answered | Held-out plans | Held-out questions | Held-out accuracy | All plans (inflated) |")
    print("|---|---|---|---|---|---|---|")
    for kind, name in KINDS.items():
        s = summary[kind]
        print(f"| {name} | {s.benchmark:,} | {s.answered:,} | {s.held_out_plans:,} | {s.held_out:,} | "
              f"{_pct(s.held_out_correct, s.held_out)} | {_pct(s.correct, s.answered)} |")
    weights = {k: benchmark[k] for k in KINDS}
    point, low, high = bootstrap(scored, weights)
    inflated = weighted_score({k: summary[k].correct / summary[k].answered for k in KINDS}, weights)
    print(f"\nCode-only score, held-out plans, question-weighted: {point:.1%} "
          f"(95% interval over plans {low:.1%} to {high:.1%})")
    print(f"Code-only score on all plans (inflated, not a result): {inflated:.1%}")
    result: dict[str, Any] = {"code_only": (point, low, high), "summary": summary}
    for name, rows in (("rooms", [r for r in scored if r.path == "rooms"]),
                       ("areas", [r for r in scored if r.path == "areas"])):
        sure = [r for r in rows if r.held_out and r.confident]
        print(f"{name}: confident on {len(sure)} of {sum(r.held_out for r in rows)} held-out questions, "
              f"{_pct(sum(r.correct for r in sure), len(sure))} correct when confident")
    if tallies is None:
        return result
    print("\n## Opus 5.5 answers in existing eval runs\n")
    print("| Kind | Prompt | Answers | Distinct questions | Correct | Accuracy | 95% interval | Mean cost |")
    print("|---|---|---|---|---|---|---|---|")
    for kind, name in KINDS.items():
        for prompt in ("plain", "conventions"):
            t = tallies.get((kind, prompt))
            if t is None:
                continue
            lo, hi = wilson(t.correct, t.answers)
            print(f"| {name} | {prompt} | {t.answers} | {len(t.questions)} | {t.correct} | "
                  f"{_pct(t.correct, t.answers)} | {lo:.0%} to {hi:.0%} | ${t.cost / t.answers:.4f} |")
    result["llm"] = llm_estimate(tallies, weights)
    result["routing"] = routing_estimate(scored, summary, tallies, weights)
    return result


def _group_of(kind: str) -> str:
    if kind.startswith("swiss:area-"):
        return "swiss:area"
    name = kind.split(":")[1]
    if name in room_questions.FINNISH_KINDS or name in room_questions.SWISS_KINDS:
        return kind.split(":")[0] + ":rooms"
    return kind.split(":")[0] + ":objects"


FEW = 30
"""Fewer distinct questions than this: the kind's LLM accuracy is called uncertain."""


def _plain_accuracy(tallies, kind: str, weights: Mapping[str, int]) -> tuple[float, float, float, bool]:
    """(estimate, low, high, uncertain) of the plain LLM on ``kind``. A kind without answers
    gets the pooled accuracy of its group (Finnish objects, Swiss objects, ...) and the range 0 to 1."""
    t = tallies.get((kind, "plain"))
    if t is None or t.answers == 0:
        group = _group_of(kind)
        pooled = [v for (k, p), v in tallies.items() if p == "plain" and _group_of(k) == group]
        hits = sum(v.correct for v in pooled)
        total = sum(v.answers for v in pooled)
        return (hits / total if total else 0.5), 0.0, 1.0, True
    lo, hi = wilson(t.correct, t.answers)
    return t.correct / t.answers, lo, hi, len(t.questions) < FEW


def llm_estimate(tallies, weights: Mapping[str, int]) -> dict[str, Any]:
    """The plain LLM's score on the whole benchmark, estimated from its per-kind accuracies."""
    point = {}
    low = {}
    high = {}
    uncertain = []
    for kind in weights:
        point[kind], low[kind], high[kind], few = _plain_accuracy(tallies, kind, weights)
        if few:
            uncertain.append(kind)
    estimate = weighted_score(point, weights)
    # The uncertain kinds at the ends of their intervals, the others at their point estimate.
    lo_range = weighted_score({k: (low[k] if k in uncertain else point[k]) for k in weights}, weights)
    hi_range = weighted_score({k: (high[k] if k in uncertain else point[k]) for k in weights}, weights)
    # Every kind at the ends of its interval.
    lo_all = weighted_score(low, weights)
    hi_all = weighted_score(high, weights)
    share = sum(weights[k] for k in uncertain) / sum(weights.values())
    print(f"\nPlain-LLM estimate (Opus 5.5, baseline context), question-weighted: {estimate:.1%}")
    print(f"  uncertain kinds (< {FEW} distinct questions measured), {share:.1%} of the benchmark: "
          + ", ".join(uncertain))
    print(f"  uncertain kinds at the ends of their 95% intervals: {lo_range:.1%} to {hi_range:.1%}")
    print(f"  every kind at the ends of its 95% interval: {lo_all:.1%} to {hi_all:.1%}")
    return {"estimate": estimate, "range": (lo_range, hi_range), "range_all": (lo_all, hi_all),
            "uncertain": uncertain, "point": point}


def _best_llm(tallies, kind: str) -> tuple[float, LlmTally | None, str]:
    for prompt in ("conventions", "plain"):
        t = tallies.get((kind, prompt))
        if t is not None and t.answers:
            return t.correct / t.answers, t, prompt
    return float("nan"), None, "not measured"


def routing_estimate(scored, summary, tallies, weights: Mapping[str, int]) -> dict[str, Any]:
    """The estimated score and LLM cost per question of ROUTING."""
    print("\n## Routing\n")
    print("| Kind | Code, held-out | Best Opus (prompt, answers) | Code on the same questions | Route | Evidence "
          "| Estimated accuracy | LLM cost per question |")
    print("|---|---|---|---|---|---|---|---|")
    by_question = {(r.plan, r.question): r for r in scored}
    accuracy, cost = {}, {}
    plain_cost_by_group: dict[str, list[float]] = defaultdict(list)
    for (kind, prompt), t in tallies.items():
        if prompt == "plain" and t.answers:
            plain_cost_by_group[_group_of(kind)].append(t.cost / t.answers)
    for kind in weights:
        s = summary[kind]
        code = s.held_out_correct / s.held_out
        best, tally, prompt = _best_llm(tallies, kind)
        if tally is not None:
            llm_cost = tally.cost / tally.answers
        else:
            costs = plain_cost_by_group.get(_group_of(kind), [])
            llm_cost = sum(costs) / len(costs) if costs else 0.0
        if math.isnan(best):
            best = _plain_accuracy(tallies, kind, weights)[0]
        route, evidence = ROUTING[kind]
        if route == "code":
            accuracy[kind], cost[kind] = code, 0.0
        elif route == "llm":
            accuracy[kind], cost[kind] = best, llm_cost
        else:
            rows = [r for r in scored if r.kind == kind and r.held_out]
            sure = [r for r in rows if r.confident]
            share = len(sure) / len(rows)
            sure_acc = sum(r.correct for r in sure) / len(sure) if sure else 0.0
            accuracy[kind] = share * sure_acc + (1 - share) * best
            cost[kind] = (1 - share) * llm_cost
        llm_text = f"{best:.1%} ({prompt}, {tally.answers})" if tally is not None else "not measured"
        same = "-"
        if tally is not None:
            # The code's answers to the distinct questions that the LLM answered: on fitted plans
            # too, so the held-out part is said.
            pairs = [by_question[key] for key in tally.questions if key in by_question]
            same = (f"{sum(r.correct for r in pairs)} of {len(pairs)} "
                    f"({sum(r.held_out for r in pairs)} held out)")
        print(f"| {KINDS[kind]} | {code:.1%} | {llm_text} | {same} | {route} | {evidence} "
              f"| {accuracy[kind]:.1%} | ${cost[kind]:.4f} |")
    score = weighted_score(accuracy, weights)
    mean_cost = weighted_score(cost, weights)
    print(f"\nRouting estimate: {score:.1%}, LLM cost ${mean_cost:.4f} per question on average")
    return {"score": score, "cost": mean_cost, "accuracy": accuracy, "cost_by_kind": cost}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset_dir", type=Path, help="the benchmark folder, e.g. data/floorplans")
    parser.add_argument("--area-cache", type=Path, default=Path("var/area-cache"), help="the plan analyses of scripts/eval_areas.py")
    parser.add_argument("--rooms", type=Path, required=True, help="output of python -m chatchtg.rooms.bench")
    parser.add_argument("--area-tuned", type=Path, nargs="*", default=[], help="files of plan stems the area pipeline was tuned on")
    parser.add_argument("--llm", type=Path, nargs="*", default=[], help="eval-run files with Opus answers")
    parser.add_argument("--out", type=Path, help="write one row per scored question")
    args = parser.parse_args(argv)

    from chatchtg.config import Settings
    from chatchtg.wiring import build_extraction_store

    dataset_dir = args.dataset_dir.resolve()
    questions = [json.loads(line) for line in (dataset_dir / "questions.jsonl").open(encoding="utf-8") if line.strip()]
    tuned: set[str] = set()
    for path in args.area_tuned:
        tuned.update(path.read_text().split())
    answers = load_object_answers(questions, build_extraction_store(Settings.from_env()))
    answers["rooms"] = load_room_answers(dataset_dir, args.rooms)
    answers["areas"] = load_area_answers(questions, args.area_cache, tuned)
    for name, found in answers.items():
        print(f"stored answers of {name}: {len(found):,}")

    scored, unclaimed, not_run = merge(questions, answers)
    benchmark = Counter()
    for row in questions:
        found = claims(row)
        benchmark[found[0].kind if found else "unclaimed"] += 1
    print(f"Questions: {len(questions):,}; claimed by exactly one path: {len(questions) - len(unclaimed):,}; "
          f"by none: {len(unclaimed):,}; by two or more: 0")
    for row in unclaimed[:20]:
        print(f"  unclaimed: {row['path']}: {row['question']}")
    missing = Counter(claim.kind for _, claim in not_run)
    missing_plans = {row["path"] for row, _ in not_run}
    print(f"Claimed but without a stored answer (not run): {len(not_run):,} questions on {len(missing_plans):,} plans "
          + (str(dict(missing)) if missing else ""))
    area_plans = {r.plan for r in scored if r.path == "areas"}
    held_area_plans = {r.plan for r in scored if r.path == "areas" and r.held_out}
    print(f"Area pipeline answers on {len(area_plans):,} plans, {len(held_area_plans):,} of them held out")
    if args.out:
        with args.out.open("w", encoding="utf-8") as out:
            for row in scored:
                out.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")
    tallies = None
    if args.llm:
        rows, used = read_runs(args.llm)
        print(f"LLM rows: {len(rows):,} from {len(used)} distinct files")
        tallies = tally_llm(rows)
    report(scored, benchmark, tallies)
    return 0


if __name__ == "__main__":
    sys.exit(main())
