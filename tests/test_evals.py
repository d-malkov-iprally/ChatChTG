import json

import pytest

from chatchtg.answering import AnswerService
from chatchtg.cli import main
from chatchtg.context.baseline import BaselineComposer
from chatchtg.context.types import ComputedAnswer, Context, TextPart
from chatchtg.evals.runner import UnknownPriceError, run_eval
from chatchtg.evals.scoring import extract_number, score
from chatchtg.llm.client import Completion
from chatchtg.llm.fake import FakeLLM
from chatchtg.llm.pricing import cost_usd
from conftest import PNG


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("There are 13 windows.", 13),
        ("Rooms 1 and 2 together: 45.5 m2", 45.5),
        ("About 1,234.5 m", 1234.5),
        ("Floor 2 has 150 windows", 150),
        ("The area is 45.5 m2.", 45.5),
        ("-3", -3),
        ("Pinta-ala on 63,84 m2", 63.84),
        ("12,5", 12.5),
        ("About 50 m^2", 50),
        ("1,234,567 mm", 1234567),
        ("No number here", None),
    ],
)
def test_extract_number_takes_the_last_number(text, expected):
    assert extract_number(text) == expected


def test_counts_must_be_exact():
    assert score("count", 13, "13 windows") is True
    assert score("count", 13, "14 windows") is False
    assert score("count", 13, "I cannot tell") is False


@pytest.mark.parametrize("question_type", ["area", "length", "volume"])
def test_measures_allow_two_percent(question_type):
    assert score(question_type, 100.0, "101.9") is True
    assert score(question_type, 100.0, "98") is True
    assert score(question_type, 100.0, "102.5") is False


def test_free_form_answers_are_not_scored():
    assert score("text", "Kitchen", "Kitchen") is None
    assert score("other", 3, "3") is None


@pytest.fixture
def dataset(data_root):
    dataset_dir = data_root / "preview"
    drawings = dataset_dir / "data" / "cubicasa5k"
    drawings.mkdir(parents=True)
    (drawings / "a.png").write_bytes(PNG)
    (drawings / "b.pdf").write_bytes(b"%PDF-1.7")
    (drawings / "LICENSE.md").write_text("CC BY-NC-SA 4.0")
    questions = [
        {"path": "data/cubicasa5k/a.png", "question": "Windows?", "answer": 4, "unit": "count", "type": "count"},
        {"path": "data/cubicasa5k/a.png", "question": "Area?", "answer": 50.0, "unit": "m2", "type": "area"},
        {"path": "data/cubicasa5k/a.png", "question": "Room name?", "answer": "Kitchen", "unit": "", "type": "text"},
        {"path": "data/cubicasa5k/b.pdf", "question": "Windows?", "answer": 4, "unit": "count", "type": "count"},
    ]
    (dataset_dir / "questions.jsonl").write_text("\n".join(json.dumps(q) for q in questions) + "\n")
    return dataset_dir


def test_run_eval_scores_raster_questions_and_writes_results(dataset, library, tmp_path):
    service = AnswerService(library, BaselineComposer(), FakeLLM(reply="The answer is 4"))
    out_path = tmp_path / "runs" / "result.jsonl"

    summary = run_eval(dataset, library, service, out_path).summary

    assert summary == {
        "count": {"total": 1, "scored": 1, "correct": 1},
        "area": {"total": 1, "scored": 1, "correct": 0},
        "text": {"total": 1, "scored": 0, "correct": 0},
    }
    rows = [json.loads(line) for line in out_path.read_text().splitlines()]
    assert [row["correct"] for row in rows] == [True, False, None]
    assert rows[0]["composer"] == "baseline@1"
    assert rows[0]["extracted_value"] == 4
    assert rows[0]["context"]["parts"][1] == {"type": "text", "text": "Windows?"}
    assert library.count() == 1


def test_run_eval_respects_the_limit(dataset, library, service, tmp_path):
    summary = run_eval(dataset, library, service, tmp_path / "result.jsonl", limit=1).summary

    assert sum(counts["total"] for counts in summary.values()) == 1


def test_cli_ingest_is_idempotent_and_eval_runs(dataset, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CHATCHTG_DATA_ROOT", str(dataset.parent))
    monkeypatch.setenv("CHATCHTG_STATE_DIR", str(tmp_path / "state"))

    assert main(["ingest", str(dataset)]) == 0
    assert "New documents: 1." in capsys.readouterr().out
    assert main(["ingest", str(dataset)]) == 0
    assert "New documents: 0." in capsys.readouterr().out

    out_path = tmp_path / "result.jsonl"
    assert main(["eval", str(dataset), "--composer", "baseline", "--out", str(out_path)]) == 0
    assert "baseline@1" in capsys.readouterr().out
    assert len(out_path.read_text().splitlines()) == 3


def test_cli_eval_without_questions_file_writes_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CHATCHTG_STATE_DIR", str(tmp_path / "state"))

    assert main(["eval", str(tmp_path / "missing")]) == 2
    assert "No questions.jsonl" in capsys.readouterr().err
    assert not (tmp_path / "state" / "eval-runs").exists()


@pytest.mark.parametrize("escape", ["../outside.png", "{absolute}"])
def test_run_eval_refuses_paths_outside_the_dataset(dataset, library, service, tmp_path, escape):
    outside = dataset.parent / "outside.png"
    outside.write_bytes(PNG + b"secret")
    question = {"path": escape.format(absolute=outside), "question": "Windows?", "answer": 4, "type": "count"}
    (dataset / "questions.jsonl").write_text(json.dumps(question) + "\n")

    summary = run_eval(dataset, library, service, tmp_path / "result.jsonl").summary

    assert summary == {}
    assert library.count() == 0


class MeteredLLM:
    """Answers "4" and reports the usage of a model with a known price."""

    def __init__(self, model="claude-opus-5-5"):
        self.model = model

    def complete(self, context):
        usage = {"input_tokens": 1_000_000, "output_tokens": 100_000}
        return Completion(text="4", model=self.model, usage=usage)


def test_cost_usd_uses_the_price_of_the_model():
    usage = {"input_tokens": 1_000_000, "output_tokens": 100_000, "cache_read_input_tokens": 500_000}

    assert cost_usd("claude-opus-5-5", usage) == pytest.approx(4.0 + 2.0 + 2.0)
    assert cost_usd("fake", usage) == 0
    assert cost_usd("unknown-model", usage) is None


def test_run_eval_records_usage_and_cost(dataset, library, tmp_path):
    service = AnswerService(library, BaselineComposer(), MeteredLLM())
    out_path = tmp_path / "result.jsonl"

    run = run_eval(dataset, library, service, out_path)

    rows = [json.loads(line) for line in out_path.read_text().splitlines()]
    assert rows[0]["usage"] == {"input_tokens": 1_000_000, "output_tokens": 100_000}
    assert rows[0]["cost_usd"] == pytest.approx(6.0)
    assert run.cost_usd == pytest.approx(18.0)
    assert run.stopped_for_cost is False


def test_run_eval_stops_before_the_cost_limit(dataset, library, tmp_path):
    service = AnswerService(library, BaselineComposer(), MeteredLLM())
    out_path = tmp_path / "result.jsonl"

    run = run_eval(dataset, library, service, out_path, max_cost=13.0)

    # Two answers cost 12 dollars. A third one would pass the limit.
    assert len(out_path.read_text().splitlines()) == 2
    assert run.cost_usd == pytest.approx(12.0)
    assert run.stopped_for_cost is True


def test_run_eval_with_a_cost_limit_needs_a_known_price(dataset, library, tmp_path):
    service = AnswerService(library, BaselineComposer(), MeteredLLM(model="unknown-model"))
    out_path = tmp_path / "result.jsonl"

    with pytest.raises(UnknownPriceError):
        run_eval(dataset, library, service, out_path, max_cost=1.0)

    assert len(out_path.read_text().splitlines()) == 1


class StatedComposer:
    """A composer whose context carries an answer that code computed and stated."""

    name = "stated"
    version = 1

    def compose(self, request):
        return Context(
            system="Not sent.",
            parts=[TextPart(text=request.message)],
            computed=ComputedAnswer(value=50.004, basis="printed", statement="Area of room 1.\n\nAnswer: 50.00 m²"),
        )


def test_a_code_stated_answer_costs_nothing_and_passes_a_cost_limit(dataset, library, tmp_path):
    # The LLM has no price: a call to it would raise UnknownPriceError under a cost limit.
    service = AnswerService(library, StatedComposer(), MeteredLLM(model="unknown-model"))
    out_path = tmp_path / "result.jsonl"

    run = run_eval(dataset, library, service, out_path, max_cost=1.0)

    rows = [json.loads(line) for line in out_path.read_text().splitlines()]
    assert cost_usd("area-pipeline", {}) == 0
    assert len(rows) == 3
    assert all(row["model"] == "area-pipeline" and row["cost_usd"] == 0 for row in rows)
    assert rows[1]["correct"] is True and rows[1]["extracted_value"] == 50.0
    assert rows[1]["context"]["computed"] == {"value": 50.004, "unit": "m2", "basis": "printed", "doubts": []}
    assert run.cost_usd == 0 and run.stopped_for_cost is False


def test_run_eval_sample_is_balanced_and_repeatable(dataset, library, service, tmp_path):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"

    run = run_eval(dataset, library, service, first, sample=2, seed=3)
    run_eval(dataset, library, service, second, sample=2, seed=3)

    rows = [json.loads(line) for line in first.read_text().splitlines()]
    assert len(rows) == 2
    assert len({row["type"] for row in rows}) == 2
    assert [row["question"] for row in rows] == [
        json.loads(line)["question"] for line in second.read_text().splitlines()
    ]
    assert sum(counts["total"] for counts in run.summary.values()) == 2


def test_cli_eval_prints_the_cost(dataset, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CHATCHTG_DATA_ROOT", str(dataset.parent))
    monkeypatch.setenv("CHATCHTG_STATE_DIR", str(tmp_path / "state"))

    assert main(["eval", str(dataset), "--sample", "2", "--max-cost", "1"]) == 0

    assert "Questions answered: 2. Cost: $0.00." in capsys.readouterr().out


def test_run_eval_resume_keeps_answers_and_asks_only_the_rest(dataset, library, tmp_path):
    service = AnswerService(library, BaselineComposer(), MeteredLLM())
    out_path = tmp_path / "result.jsonl"
    run_eval(dataset, library, service, out_path, limit=1)
    first_line = out_path.read_text()

    run = run_eval(dataset, library, service, out_path, resume=True)

    lines = out_path.read_text().splitlines(keepends=True)
    assert lines[0] == first_line
    assert [json.loads(line)["question"] for line in lines] == ["Windows?", "Area?", "Room name?"]
    assert sum(counts["total"] for counts in run.summary.values()) == 3
    assert run.cost_usd == pytest.approx(18.0)


def test_run_eval_resume_drops_a_line_that_was_cut_off(dataset, library, tmp_path):
    out_path = tmp_path / "result.jsonl"
    service = AnswerService(library, BaselineComposer(), FakeLLM(reply="The answer is 4"))
    run_eval(dataset, library, service, out_path)
    complete = out_path.read_text().splitlines(keepends=True)
    out_path.write_text("".join(complete[:2]) + complete[2][:20])

    run = run_eval(dataset, library, service, out_path, resume=True)

    rows = [json.loads(line) for line in out_path.read_text().splitlines()]
    assert [row["question"] for row in rows] == ["Windows?", "Area?", "Room name?"]
    assert sum(counts["total"] for counts in run.summary.values()) == 3


def test_run_eval_counts_the_questions_it_skips(dataset, library, service, tmp_path):
    question = {"path": "data/cubicasa5k/gone.png", "question": "Windows?", "answer": 4, "type": "count"}
    with (dataset / "questions.jsonl").open("a") as out:
        out.write(json.dumps(question) + "\n")

    run = run_eval(dataset, library, service, tmp_path / "result.jsonl")

    assert run.skipped == 1
    assert sum(counts["total"] for counts in run.summary.values()) == 3


def test_cli_eval_refuses_resume_without_out(dataset, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CHATCHTG_STATE_DIR", str(tmp_path / "state"))

    assert main(["eval", str(dataset), "--resume"]) == 2
    assert "--resume needs --out" in capsys.readouterr().err
    assert not (tmp_path / "state" / "eval-runs").exists()


def test_cli_eval_reports_an_unknown_setting(dataset, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CHATCHTG_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("CHATCHTG_ROOM_COUNTER", "sometimes")

    assert main(["eval", str(dataset)]) == 2
    assert "Unknown room counter mode" in capsys.readouterr().err
