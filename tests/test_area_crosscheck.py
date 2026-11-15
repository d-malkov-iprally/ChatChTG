"""The policy of the area cross-check: when a second estimate replaces a measured area."""

import importlib.util
from pathlib import Path

import pytest

from chatchtg.areas.crosscheck import compare, eligible
from chatchtg.areas.pipeline import PlanAnalysis, PlanRoom
from chatchtg.context.areas import GAP_DOUBT, SCALE_DOUBT


def test_a_difference_above_ten_percent_is_a_disagreement():
    check = compare(100.0, 111.0)

    assert check.verdict == "disagree" and check.chosen == 111.0
    assert check.difference == pytest.approx(0.11)


def test_a_difference_of_ten_percent_or_less_keeps_the_tool_value():
    assert compare(100.0, 110.0).verdict == "agree"
    assert compare(100.0, 91.0).chosen == 100.0


def test_no_usable_estimate_keeps_the_tool_value():
    for estimate in (None, 0.0, -3.0):
        check = compare(100.0, estimate)

        assert check.verdict == "no-estimate" and check.chosen == 100.0 and check.difference is None


def test_the_threshold_can_be_set():
    assert compare(100.0, 106.0, threshold=0.05).verdict == "disagree"
    assert compare(100.0, 106.0, threshold=0.08).verdict == "agree"


def test_only_a_doubtful_measured_answer_is_eligible():
    assert eligible("measured", [SCALE_DOUBT])
    assert not eligible("measured", [])
    assert not eligible("printed", [GAP_DOUBT])
    assert not eligible(None, [GAP_DOUBT])


# The replay script: its counting decides the gates of the benchmark record.

_SPEC = importlib.util.spec_from_file_location(
    "eval_area_check", Path(__file__).parent.parent / "scripts" / "eval_area_check.py"
)
script = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(script)


def _question(number, answer, path="data/swiss_dwellings/a.png"):
    return {
        "path": path, "question": f"What is the net floor area of room {number} ('Zimmer') in m²?",
        "answer": answer, "type": "area", "unit": "m2",
    }


def _plan(scale=200, missing=()):
    rooms = [
        PlanRoom(
            number=number, name="Zimmer", area=area, method=method, printed_area=None,
            measured_area=area, apartment="A", outdoor=False, seed=(0.0, 0.0),
        )
        for number, area, method in (("1.01", 10.0, "measured"), ("1.02", 20.0, "printed"))
    ]
    return PlanAnalysis(
        scale=scale, scale_read=True, pixels_per_metre=29.5, rooms=rooms, missing_numbers=list(missing)
    )


def test_a_signal_row_says_whether_the_check_would_be_made():
    rows = script.signal_rows([_question("1.01", 12.5), _question("1.02", 20.0), _question("1.09", 5.0)], _plan())

    measured, printed, missing = rows
    assert measured["eligible"] and measured["basis"] == "measured" and measured["doubts"] == [SCALE_DOUBT]
    assert measured["error"] == pytest.approx(-0.2)
    assert not printed["eligible"] and printed["basis"] == "printed"
    assert not missing["confident"] and missing["value"] is None and missing["error"] is None


def test_a_clean_plan_at_one_to_hundred_has_no_eligible_question():
    rows = script.signal_rows([_question("1.01", 10.0)], _plan(scale=100))

    assert rows[0]["confident"] and not rows[0]["eligible"]


def test_a_plan_the_pipeline_could_not_analyse_has_no_values():
    rows = script.signal_rows([_question("1.01", 10.0)], None)

    assert rows[0]["value"] is None and not rows[0]["eligible"]


def test_the_signal_table_counts_wrong_and_gross_answers():
    rows = script.signal_rows([_question("1.01", 12.5), _question("1.02", 20.0)], _plan())
    rows += script.signal_rows([_question("1.01", 10.1, path="data/swiss_dwellings/b.png")], _plan(scale=100))

    everything, chosen, others = script.signal_table(rows)

    assert (everything["questions"], everything["wrong"], everything["gross"], everything["plans"]) == (2, 1, 1, 2)
    assert (chosen["questions"], chosen["gross"]) == (1, 1)
    assert (others["questions"], others["wrong"]) == (1, 0)


def test_select_takes_one_eligible_question_per_plan():
    rows = []
    for name in "abc":
        path = f"data/swiss_dwellings/{name}.png"
        rows += script.signal_rows([_question("1.01", 10.0, path), _question("1.02", 20.0, path)], _plan())

    questions, plans = script.select_questions(rows, n=2, seed=0)

    assert plans == 3 and len(questions) == 2
    assert len({question["path"] for question in questions}) == 2
    assert all("1.01" in question["question"] for question in questions)  # 1.02 is printed
    assert set(questions[0]) == {"path", "question", "answer", "type", "unit"}
    assert questions == script.select_questions(rows, n=2, seed=0)[0]


def _run_row(path, estimate, expected):
    question = _question("1.01", expected, path)
    return {
        "path": path, "question": question["question"], "expected": expected,
        "extracted_value": estimate, "cost_usd": 0.05,
    }


def test_the_replay_counts_what_the_check_would_change():
    signals = []
    run = []
    # the pipeline says 10.0 for each; (reference, the LLM's estimate)
    cases = {"a": (12.5, 12.4), "b": (10.0, 12.0), "c": (10.0, 10.5), "d": (12.5, None), "e": (12.5, 10.2)}
    for name, (expected, estimate) in cases.items():
        path = f"data/swiss_dwellings/{name}.png"
        signals += script.signal_rows([_question("1.01", expected, path)], _plan())
        run.append(_run_row(path, estimate, expected))
    run.append(_run_row("data/swiss_dwellings/unknown.png", 10.0, 10.0))  # no signal row

    rows = script.replay_rows(signals, run)
    counts = script.replay_counts(rows)

    assert counts == {
        "questions": 5, "tool_correct": 2, "llm_correct": 1, "check_correct": 2, "no_estimate": 1,
        "flagged": 2, "tool_wrong_among_flagged": 1, "became_right": 1, "became_wrong": 1,
    }
    assert [row["verdict"] for row in rows] == ["disagree", "disagree", "agree", "no-estimate", "agree"]
    assert rows[0]["chosen"] == 12.4 and rows[3]["chosen"] == 10.0
    assert script.replay_counts(script.replay_rows(signals, run, threshold=0.25))["flagged"] == 0
