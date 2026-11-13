import json

import pytest

from chatchtg.answering import AnswerService
from chatchtg.cli import main
from chatchtg.context.baseline import BaselineComposer
from chatchtg.evals.runner import run_eval
from chatchtg.evals.scoring import extract_number, score
from chatchtg.llm.fake import FakeLLM
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

    summary = run_eval(dataset, library, service, out_path)

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
    summary = run_eval(dataset, library, service, tmp_path / "result.jsonl", limit=1)

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
