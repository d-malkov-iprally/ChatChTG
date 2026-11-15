"""Runs the dataset questions through the answer service and scores the answers."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from chatchtg.answering import AnswerService
from chatchtg.documents.library import RASTER_SUFFIXES, DocumentLibrary, UnsupportedDocumentError
from chatchtg.evals.benchmark import sample_subset
from chatchtg.evals.judge import Judge
from chatchtg.evals.scoring import extract_number, score, score_value
from chatchtg.llm.pricing import cost_usd
from chatchtg.schemas import AnswerRequest


class UnknownPriceError(RuntimeError):
    """A cost limit was set, and a model without a known price answered."""


@dataclass
class EvalRun:
    """What one eval run did."""

    summary: dict[str, dict[str, int]] = field(default_factory=dict)
    """Counts per question type: total, scored, correct."""
    cost_usd: float = 0.0
    """Cost of the answers and of the judge, for the models that have a known price."""
    stopped_for_cost: bool = False
    """True when the run ended early because the next question could pass ``max_cost``."""
    skipped: int = 0
    """Questions left out because their drawing is missing, unreadable or outside the dataset."""


def read_questions(
    dataset_dir: Path, questions_file: str = "questions.jsonl"
) -> Iterator[dict[str, Any]]:
    """Yield the questions of ``questions_file`` whose drawing is a raster image."""
    with (dataset_dir / questions_file).open(encoding="utf-8") as lines:
        for line in lines:
            if line.strip():
                question = json.loads(line)
                if Path(question["path"]).suffix.lower() in RASTER_SUFFIXES:
                    yield question


def _register(library: DocumentLibrary, dataset_dir: Path, path: str) -> str | None:
    """Register a drawing named by questions.jsonl. Return None when it cannot be used.

    questions.jsonl is downloaded data, so a path that leaves the dataset folder is refused.
    """
    root = dataset_dir.resolve()
    target = (root / path).resolve()
    if not target.is_relative_to(root):
        return None
    try:
        return library.register_file(target).id
    except (UnsupportedDocumentError, OSError):  # missing, a folder, or not readable
        return None


def run_eval(
    dataset_dir: Path,
    library: DocumentLibrary,
    service: AnswerService,
    out_path: Path,
    limit: int | None = None,
    questions_file: str = "questions.jsonl",
    sample: int | None = None,
    seed: int = 0,
    max_cost: float | None = None,
    resume: bool = False,
    judge: Judge | None = None,
) -> EvalRun:
    """Write one JSON line per question to ``out_path``. Return the counts and the cost.

    ``questions_file`` names the question file inside ``dataset_dir``.
    ``sample`` picks that many questions, balanced over source and question type; the same
    ``seed`` gives the same questions.
    ``max_cost`` is a limit in US dollars, for the answers and the judge together. The run
    stops before a question when the cost so far plus the most expensive question so far
    would pass it. One question that costs more than every question before it can still
    pass the limit.
    ``resume`` continues a run that was interrupted: the answers already in ``out_path``
    are kept and counted, and their questions are not asked again. A last line that the
    interruption cut off is removed, and its question is asked again.

    With a ``judge``, each answer that the rule can score is also read by the judge. The row
    gets ``judge_*`` fields and the counts get ``judged``, ``judge_correct`` and ``disagree``.
    ``correct`` stays the result of the rule.
    """
    questions: Iterator[dict[str, Any]] | list[dict[str, Any]]
    questions = read_questions(dataset_dir, questions_file)
    if sample is not None:
        questions = sample_subset(list(questions), sample, seed)

    zero = {"total": 0, "scored": 0, "correct": 0}
    if judge is not None:
        zero |= {"judged": 0, "judge_correct": 0, "disagree": 0}
    run = EvalRun()
    summary = run.summary
    document_ids: dict[str, str | None] = {}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = 0
    largest_cost = 0.0
    answered: set[tuple[str, str]] = set()
    if resume and out_path.is_file():
        for row in _read_complete_rows(out_path):
            answered.add((row["path"], row["question"]))
            _count(summary, zero, row["type"], row["correct"], row.get("judge_correct"))
            row_cost = (row.get("cost_usd") or 0.0) + (row.get("judge_cost_usd") or 0.0)
            run.cost_usd += row_cost
            largest_cost = max(largest_cost, row_cost)
        done = len(answered)
    with out_path.open("a" if answered else "w", encoding="utf-8") as out:
        for question in questions:
            if (question["path"], question["question"]) in answered:
                continue
            if limit is not None and done >= limit:
                break
            if max_cost is not None and run.cost_usd + largest_cost > max_cost:
                run.stopped_for_cost = True
                break
            path = question["path"]
            if path not in document_ids:
                document_ids[path] = _register(library, dataset_dir, path)
            document_id = document_ids[path]
            if document_id is None:
                run.skipped += 1
                continue

            trace = service.answer_traced(
                AnswerRequest(document_id=document_id, message=question["question"])
            )
            text = trace.response.text
            cost = cost_usd(trace.completion.model, trace.completion.usage)
            correct = score(question["type"], question["answer"], text)
            row = {
                "path": path,
                "document_id": document_id,
                "question": question["question"],
                "type": question["type"],
                "unit": question.get("unit"),
                "expected": question["answer"],
                "composer": trace.response.composer,
                "model": trace.response.model,
                "context": trace.context.summary(),
                "response": text,
                "extracted_value": extract_number(text),
                "correct": correct,
                "usage": trace.completion.usage,
                "cost_usd": cost,
            }
            costs = [(trace.completion.model, cost)]
            judge_correct = None
            if judge is not None and correct is not None:
                judgement = judge.read(question["question"], question.get("unit"), text)
                reading = judgement.reading
                if reading is not None:
                    judge_correct = score_value(question["type"], question["answer"], reading.value)
                row |= {
                    "judge_model": judgement.model,
                    "judge_value": reading.value if reading else None,
                    "judge_unit": reading.unit if reading else None,
                    "judge_quote": reading.quote if reading else None,
                    "judge_correct": judge_correct,
                    "judge_error": judgement.error,
                    "judge_usage": judgement.usage,
                    "judge_cost_usd": cost_usd(judgement.model, judgement.usage),
                }
                costs.append((judgement.model, row["judge_cost_usd"]))
            if trace.response.context_composer is not None:
                row["context_composer"] = trace.response.context_composer
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()  # keep the answers already paid for if the run is interrupted
            if max_cost is not None:
                for model, model_cost in costs:
                    if model_cost is None:
                        raise UnknownPriceError(
                            f"No price for model {model!r}, so the cost limit cannot be "
                            "applied. Add the model to chatchtg/llm/pricing.py."
                        )
            question_cost = sum(model_cost or 0.0 for _, model_cost in costs)
            run.cost_usd += question_cost
            largest_cost = max(largest_cost, question_cost)

            _count(summary, zero, question["type"], correct, judge_correct)
            done += 1
    return run


def _count(
    summary: dict[str, dict[str, int]],
    zero: dict[str, int],
    question_type: str,
    correct: bool | None,
    judge_correct: bool | None,
) -> None:
    """Add one answer to the counts of its question type. ``zero`` is an empty set of counts."""
    counts = summary.setdefault(question_type, dict(zero))
    counts["total"] += 1
    if correct is not None:
        counts["scored"] += 1
        counts["correct"] += int(correct)
    if judge_correct is not None and "judged" in counts:
        counts["judged"] += 1
        counts["judge_correct"] += int(judge_correct)
        counts["disagree"] += int(judge_correct != correct)


def _read_complete_rows(path: Path) -> list[dict[str, Any]]:
    """Return the rows of an earlier run. Remove a last line that was cut off mid-write.

    Without the removal, the cut-off line could not be parsed, and the next row would be
    appended to it.
    """
    data = path.read_bytes()
    end = data.rfind(b"\n") + 1
    if end != len(data):
        with path.open("r+b") as out:
            out.truncate(end)
    # Not splitlines(): it also splits on separators that a JSON string may contain.
    return [json.loads(line) for line in data[:end].decode("utf-8").split("\n") if line.strip()]
