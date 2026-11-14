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
from chatchtg.evals.scoring import extract_number, score
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
    """Cost of the answers whose model has a known price."""
    stopped_for_cost: bool = False
    """True when the run ended early because the next question could pass ``max_cost``."""


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
) -> EvalRun:
    """Write one JSON line per question to ``out_path``. Return the counts and the cost.

    ``questions_file`` names the question file inside ``dataset_dir``.
    ``sample`` picks that many questions, balanced over source and question type; the same
    ``seed`` gives the same questions.
    ``max_cost`` is a limit in US dollars. The run stops before a question when the cost
    so far plus the most expensive answer so far would pass it. One answer that costs more
    than every answer before it can still pass the limit.
    """
    questions: Iterator[dict[str, Any]] | list[dict[str, Any]]
    questions = read_questions(dataset_dir, questions_file)
    if sample is not None:
        questions = sample_subset(list(questions), sample, seed)

    run = EvalRun()
    summary = run.summary
    document_ids: dict[str, str | None] = {}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = 0
    largest_cost = 0.0
    with out_path.open("w", encoding="utf-8") as out:
        for question in questions:
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
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()  # keep the answers already paid for if the run is interrupted
            if cost is None and max_cost is not None:
                raise UnknownPriceError(
                    f"No price for model {trace.completion.model!r}, so the cost limit cannot "
                    "be applied. Add the model to chatchtg/llm/pricing.py."
                )
            run.cost_usd += cost or 0.0
            largest_cost = max(largest_cost, cost or 0.0)

            counts = summary.setdefault(question["type"], {"total": 0, "scored": 0, "correct": 0})
            counts["total"] += 1
            if correct is not None:
                counts["scored"] += 1
                counts["correct"] += int(correct)
            done += 1
    return run
