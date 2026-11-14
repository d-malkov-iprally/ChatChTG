import json
from types import SimpleNamespace

import pytest

from chatchtg.answering import AnswerService
from chatchtg.cli import main
from chatchtg.config import Settings
from chatchtg.context.baseline import BaselineComposer
from chatchtg.evals.judge import DEFAULT_JUDGE_MODEL, ClaudeJudge, FakeJudge, Reading
from chatchtg.evals.runner import run_eval
from chatchtg.evals.scoring import score_value
from chatchtg.llm.fake import FakeLLM
from chatchtg.wiring import build_judge
from conftest import PNG


def test_score_value_applies_the_dataset_rules():
    assert score_value("count", 13, 13.0) is True
    assert score_value("count", 13, 14.0) is False
    assert score_value("area", 100.0, 101.9) is True
    assert score_value("area", 100.0, 102.5) is False
    assert score_value("count", 13, None) is False
    assert score_value("text", "Kitchen", 1.0) is None
    assert score_value("other", 3, 3.0) is None


def _response(**overrides):
    fields = {
        "parsed_output": Reading(value=13, unit="windows", quote="13 windows"),
        "model": "claude-haiku-5-5",
        "stop_reason": "end_turn",
        "usage": SimpleNamespace(input_tokens=200, output_tokens=20),
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


class StubClient:
    """Stands in for anthropic.Anthropic: records the request, returns a canned response."""

    def __init__(self, response=None, error=None):
        self.requests = []
        self._response = response or _response()
        self._error = error
        self.messages = SimpleNamespace(parse=self._parse)

    def _parse(self, **request):
        self.requests.append(request)
        if self._error:
            raise self._error
        return self._response


def _read(judge):
    return judge.read("How many windows?", "count", "I see 13 windows, 2 of them on floor 1.")


def test_request_asks_haiku_for_a_reading_at_low_effort():
    client = StubClient()
    _read(ClaudeJudge(client=client))

    (request,) = client.requests
    assert DEFAULT_JUDGE_MODEL == "claude-haiku-5-5"
    assert request["model"] == "claude-haiku-5-5"
    assert request["output_format"] is Reading
    assert request["output_config"] == {"effort": "low"}
    assert request["max_tokens"] > 0
    assert request["system"]
    (message,) = request["messages"]
    assert "How many windows?" in message["content"]
    assert "2 of them on floor 1" in message["content"]
    for absent in ("fallbacks", "betas", "temperature", "thinking"):
        assert absent not in request


def test_judgement_reports_reading_served_model_and_usage():
    judgement = _read(ClaudeJudge(client=StubClient()))

    assert judgement.reading == Reading(value=13, unit="windows", quote="13 windows")
    assert judgement.model == "claude-haiku-5-5"
    assert judgement.usage == {"input_tokens": 200, "output_tokens": 20}
    assert judgement.error is None


@pytest.mark.parametrize("stop_reason", ["refusal", "max_tokens"])
def test_an_unfinished_response_is_an_error_not_a_reading(stop_reason):
    response = _response(stop_reason=stop_reason, parsed_output=None)
    judgement = _read(ClaudeJudge(client=StubClient(response)))

    assert judgement.reading is None
    assert stop_reason in judgement.error
    assert judgement.usage == {"input_tokens": 200, "output_tokens": 20}


def test_a_response_without_parsed_output_is_an_error():
    judgement = _read(ClaudeJudge(client=StubClient(_response(parsed_output=None))))

    assert judgement.reading is None
    assert judgement.error


def test_an_api_error_is_an_error_not_a_reading():
    import anthropic

    error = anthropic.APIConnectionError(message="connection lost", request=None)
    judgement = _read(ClaudeJudge(client=StubClient(error=error)))

    assert judgement.reading is None
    assert "APIConnectionError" in judgement.error
    assert judgement.model == "claude-haiku-5-5"


def test_project_key_goes_to_the_anthropic_client(monkeypatch):
    import anthropic

    created = []
    monkeypatch.setattr(anthropic, "Anthropic", lambda **kwargs: created.append(kwargs))

    ClaudeJudge(api_key="project-key")._get_client()
    ClaudeJudge()._get_client()

    assert created == [{"api_key": "project-key"}, {}]


def test_build_judge_follows_the_provider_and_the_model_setting():
    assert isinstance(build_judge(Settings.from_env({})), FakeJudge)

    default = build_judge(Settings.from_env({"CHATCHTG_LLM_PROVIDER": "claude"}))
    assert isinstance(default, ClaudeJudge)
    assert default.model == "claude-haiku-5-5"

    env = {"CHATCHTG_LLM_PROVIDER": "claude", "CHATCHTG_JUDGE_MODEL": "claude-sonnet-5-5"}
    assert build_judge(Settings.from_env(env)).model == "claude-sonnet-5-5"


@pytest.fixture
def dataset(data_root):
    dataset_dir = data_root / "preview"
    drawings = dataset_dir / "data" / "cubicasa5k"
    drawings.mkdir(parents=True)
    (drawings / "a.png").write_bytes(PNG)
    path = "data/cubicasa5k/a.png"
    questions = [
        {"path": path, "question": "Windows?", "answer": 4, "unit": "count", "type": "count"},
        {"path": path, "question": "Area?", "answer": 50.0, "unit": "m2", "type": "area"},
        {"path": path, "question": "Room name?", "answer": "Kitchen", "unit": "", "type": "text"},
    ]
    (dataset_dir / "questions.jsonl").write_text("\n".join(json.dumps(q) for q in questions) + "\n")
    return dataset_dir


class RecordingJudge(FakeJudge):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calls = []

    def read(self, question, unit, response):
        self.calls.append((question, unit, response))
        return super().read(question, unit, response)


def test_run_eval_adds_the_judge_result_next_to_the_rule(dataset, library, tmp_path):
    # The rule reads the last number (2). The judge reads the committed value (4).
    service = AnswerService(library, BaselineComposer(), FakeLLM(reply="4 windows, 2 on floor 1"))
    judge = RecordingJudge(reader=lambda _: 4.0)
    out_path = tmp_path / "result.jsonl"

    summary = run_eval(dataset, library, service, out_path, judge=judge)

    assert summary == {
        "count": {"total": 1, "scored": 1, "correct": 0, "judged": 1, "judge_correct": 1, "disagree": 1},
        "area": {"total": 1, "scored": 1, "correct": 0, "judged": 1, "judge_correct": 0, "disagree": 0},
        "text": {"total": 1, "scored": 0, "correct": 0, "judged": 0, "judge_correct": 0, "disagree": 0},
    }
    count, area, text = [json.loads(line) for line in out_path.read_text().splitlines()]
    assert count["correct"] is False
    assert count["judge_correct"] is True
    assert count["judge_value"] == 4
    assert count["judge_model"] == "fake-judge"
    assert count["judge_error"] is None
    assert {"judge_unit", "judge_quote", "judge_usage"} <= count.keys()
    assert area["judge_correct"] is False
    # A question the rule cannot score is not sent to the judge.
    assert "judge_correct" not in text
    assert judge.calls == [
        ("Windows?", "count", "4 windows, 2 on floor 1"),
        ("Area?", "m2", "4 windows, 2 on floor 1"),
    ]


def test_a_judge_error_is_not_counted_as_a_wrong_answer(dataset, library, tmp_path):
    service = AnswerService(library, BaselineComposer(), FakeLLM(reply="The answer is 4"))
    out_path = tmp_path / "result.jsonl"

    summary = run_eval(dataset, library, service, out_path, judge=FakeJudge(error="refusal"))

    assert summary["count"] == {
        "total": 1, "scored": 1, "correct": 1, "judged": 0, "judge_correct": 0, "disagree": 0
    }
    row = json.loads(out_path.read_text().splitlines()[0])
    assert row["judge_correct"] is None
    assert row["judge_error"] == "refusal"


def test_run_eval_without_a_judge_writes_no_judge_fields(dataset, library, service, tmp_path):
    out_path = tmp_path / "result.jsonl"

    summary = run_eval(dataset, library, service, out_path)

    assert summary["count"] == {"total": 1, "scored": 1, "correct": 0}
    assert not any("judge" in key for key in json.loads(out_path.read_text().splitlines()[0]))


def test_cli_eval_with_judge_prints_the_judge_columns(dataset, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CHATCHTG_DATA_ROOT", str(dataset.parent))
    monkeypatch.setenv("CHATCHTG_STATE_DIR", str(tmp_path / "state"))
    out_path = tmp_path / "result.jsonl"

    assert main(["eval", str(dataset), "--judge", "--out", str(out_path)]) == 0

    out = capsys.readouterr().out
    assert "Judge: fake-judge" in out
    assert "disagree" in out
    assert "judge_correct" in json.loads(out_path.read_text().splitlines()[0])


def test_cli_eval_without_judge_prints_no_judge_columns(dataset, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CHATCHTG_DATA_ROOT", str(dataset.parent))
    monkeypatch.setenv("CHATCHTG_STATE_DIR", str(tmp_path / "state"))

    assert main(["eval", str(dataset), "--out", str(tmp_path / "result.jsonl")]) == 0

    assert "disagree" not in capsys.readouterr().out
