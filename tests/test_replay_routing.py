import importlib.util
import json
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "replay_routing", Path(__file__).parent.parent / "scripts" / "replay_routing.py"
)
replay_routing = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(replay_routing)


def _row(question: str, model: str, correct: bool) -> dict:
    return {"path": "p.png", "question": question, "model": model, "correct": correct}


def test_replay_takes_code_answers_from_the_routing_and_llm_answers_from_the_earlier_run():
    routing = {
        ("p.png", q): _row(q, model, correct)
        for q, model, correct in [("a", "object-counter", True), ("b", "area-pipeline", False), ("c", "fake", False)]
    }
    llm = {("p.png", q): _row(q, "claude-opus-5-5", correct) for q, correct in [("a", False), ("b", True), ("c", True)]}

    result = replay_routing.replay(routing, llm)

    assert result["routing_right"] == 2
    assert result["plain_llm_right"] == 2
    assert (result["code_answered"], result["code_right"]) == (2, 1)
    assert (result["llm_answered"], result["llm_right"]) == (1, 1)
    assert (result["fixed"], result["broken"]) == (1, 1)


def test_sign_test_is_two_sided():
    assert replay_routing.sign_test(0, 0) == 1.0
    assert round(replay_routing.sign_test(15, 2), 4) == 0.0023


def test_the_plain_score_counts_only_the_questions_of_the_routing_run():
    routing = {("p.png", "a"): _row("a", "fake", False)}
    llm = {("p.png", q): _row(q, "claude-opus-5-5", True) for q in ("a", "b", "c")}

    assert replay_routing.replay(routing, llm)["plain_llm_right"] == 1


def test_a_question_that_the_earlier_run_lacks_is_refused():
    routing = {("p.png", "a"): _row("a", "fake", False)}

    with pytest.raises(ValueError, match="no answer for 1 questions"):
        replay_routing.replay(routing, {})


def test_a_run_file_with_a_question_twice_is_refused(tmp_path):
    run = tmp_path / "run.jsonl"
    run.write_text("".join(json.dumps(_row("a", "fake", ok)) + "\n" for ok in (True, False)))

    with pytest.raises(ValueError, match="twice"):
        replay_routing.load(run)
