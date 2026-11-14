import base64
from types import SimpleNamespace

import pytest
from conftest import PNG

from chatchtg.config import Settings
from chatchtg.context.types import Context, ImagePart, TextPart
from chatchtg.llm.claude import DEFAULT_MODEL, ClaudeLLM
from chatchtg.llm.client import build_llm


def _response(**overrides):
    fields = {
        "content": [SimpleNamespace(type="text", text="There are four doors.")],
        "model": "claude-opus-5-5",
        "stop_reason": "end_turn",
        "usage": SimpleNamespace(
            input_tokens=120,
            output_tokens=9,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=None,
            service_tier="standard",
        ),
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


class StubClient:
    """Stands in for anthropic.Anthropic: records the request, returns a canned response."""

    def __init__(self, response=None):
        self.requests = []
        self._response = response or _response()
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **request):
        self.requests.append(request)
        return self._response


CONTEXT = Context(
    system="You answer questions about drawings.",
    parts=[ImagePart(media_type="image/png", data=PNG), TextPart(text="How many doors?")],
)


def test_request_carries_the_context_in_order():
    client = StubClient()
    ClaudeLLM("claude-haiku-5-5", client=client).complete(CONTEXT)

    (request,) = client.requests
    assert request["model"] == "claude-haiku-5-5"
    assert request["system"] == "You answer questions about drawings."
    assert request["messages"] == [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64.standard_b64encode(PNG).decode("ascii"),
                    },
                },
                {"type": "text", "text": "How many doors?"},
            ],
        }
    ]
    assert request["max_tokens"] > 0
    assert "thinking" not in request
    assert "temperature" not in request


def test_completion_reports_text_served_model_and_usage():
    response = _response(
        content=[
            SimpleNamespace(type="thinking", thinking=""),
            SimpleNamespace(type="text", text="There are "),
            SimpleNamespace(type="text", text="four doors."),
        ],
        model="claude-opus-4-8",
    )
    completion = ClaudeLLM(DEFAULT_MODEL, client=StubClient(response)).complete(CONTEXT)

    assert completion.text == "There are four doors."
    assert completion.model == "claude-opus-4-8"
    assert completion.usage == {
        "input_tokens": 120,
        "output_tokens": 9,
        "cache_read_input_tokens": 0,
    }


def test_refusal_becomes_a_digit_free_answer():
    response = _response(content=[], stop_reason="refusal")
    completion = ClaudeLLM(DEFAULT_MODEL, client=StubClient(response)).complete(CONTEXT)

    assert "declined" in completion.text
    assert not any(ch.isdigit() for ch in completion.text)


def test_build_llm_returns_claude_without_needing_credentials(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    default = build_llm("claude")
    assert isinstance(default, ClaudeLLM)
    assert default.model == DEFAULT_MODEL
    assert build_llm("claude", "claude-haiku-5-5").model == "claude-haiku-5-5"


def test_unknown_provider_error_lists_claude():
    with pytest.raises(ValueError, match="claude"):
        build_llm("nope")


def test_settings_read_the_model():
    assert Settings.from_env({}).llm_model is None
    assert Settings.from_env({"CHATCHTG_LLM_MODEL": "claude-haiku-5-5"}).llm_model == (
        "claude-haiku-5-5"
    )
