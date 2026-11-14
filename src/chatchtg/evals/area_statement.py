"""Counts for the area answers that code stated, read from the rows of an eval run."""

from __future__ import annotations

from typing import Any

from chatchtg.evals.scoring import score_value

STATED_MODEL = "area-pipeline"
"""The model name of a code-stated area answer (``chatchtg.answering.AREA_PIPELINE_MODEL``)."""


def _key(row: dict[str, Any]) -> tuple[str, str]:
    return row["path"], row["question"]


def report(rows: list[dict[str, Any]], compare: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Counts for the area rows of an eval run.

    stated:         rows with the model ``area-pipeline``
    stated_correct: of those, ``row["correct"]`` is True (the rule on the code-made text)
    value_correct:  of those, the pipeline's own unrounded value is correct by the rule
    text_lost_the_value: the stated rows whose value is correct and whose text is not
    not_answered:   area rows with another model. In a run with the fake provider they are
                    not results.

    With ``compare`` (the rows of a run on the same questions), for the stated rows only:
    compare_correct, compare_cost_usd, compare_missing (stated rows that the other run
    does not have), gained (correct here and not there) and lost (correct there and not here).
    """
    area = [row for row in rows if row["type"] == "area"]
    stated = [row for row in area if row["model"] == STATED_MODEL]
    value_right = [
        row
        for row in stated
        if score_value("area", row["expected"], row["context"]["computed"]["value"]) is True
    ]
    result: dict[str, Any] = {
        "stated": len(stated),
        "stated_correct": sum(row["correct"] is True for row in stated),
        "value_correct": len(value_right),
        "text_lost_the_value": [row for row in value_right if row["correct"] is not True],
        "not_answered": len(area) - len(stated),
    }
    if compare is None:
        return result
    others = {_key(row): row for row in compare}
    pairs = [(row, others[_key(row)]) for row in stated if _key(row) in others]
    result |= {
        "compare_correct": sum(other["correct"] is True for _, other in pairs),
        "compare_cost_usd": sum(other.get("cost_usd") or 0.0 for _, other in pairs),
        "compare_missing": len(stated) - len(pairs),
        "gained": [row for row, other in pairs if row["correct"] is True and other["correct"] is not True],
        "lost": [row for row, other in pairs if row["correct"] is not True and other["correct"] is True],
    }
    return result
