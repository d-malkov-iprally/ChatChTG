import importlib.util
from pathlib import Path

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
