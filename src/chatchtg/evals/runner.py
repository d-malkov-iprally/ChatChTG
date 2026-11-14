"""Runs the dataset questions through the answer service and scores the answers."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from chatchtg.answering import AnswerService
from chatchtg.documents.library import RASTER_SUFFIXES, DocumentLibrary, UnsupportedDocumentError
from chatchtg.evals.judge import Judge
from chatchtg.evals.scoring import extract_number, score, score_value
from chatchtg.schemas import AnswerRequest


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
    judge: Judge | None = None,
) -> dict[str, dict[str, int]]:
    """Write one JSON line per question to ``out_path``. Return counts per question type.

    ``questions_file`` names the question file inside ``dataset_dir``.

    With a ``judge``, each answer that the rule can score is also read by the judge. The row
    gets ``judge_*`` fields and the counts get ``judged``, ``judge_correct`` and ``disagree``.
    ``correct`` stays the result of the rule.
    """
    zero = {"total": 0, "scored": 0, "correct": 0}
    if judge is not None:
        zero |= {"judged": 0, "judge_correct": 0, "disagree": 0}
    summary: dict[str, dict[str, int]] = {}
    document_ids: dict[str, str | None] = {}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = 0
    with out_path.open("w", encoding="utf-8") as out:
        for question in read_questions(dataset_dir, questions_file):
            if limit is not None and done >= limit:
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
            }
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
                }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

            counts = summary.setdefault(question["type"], dict(zero))
            counts["total"] += 1
            if correct is not None:
                counts["scored"] += 1
                counts["correct"] += int(correct)
            if judge_correct is not None:
                counts["judged"] += 1
                counts["judge_correct"] += int(judge_correct)
                counts["disagree"] += int(judge_correct != correct)
            done += 1
    return summary
