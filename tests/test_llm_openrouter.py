import base64
import io
import json
import urllib.error
import urllib.request

import pytest
from conftest import PNG

from chatchtg.config import Settings
from chatchtg.context.types import Context, ImagePart, TextPart
from chatchtg.llm.client import build_llm
from chatchtg.llm.openrouter import DEFAULT_MODEL, OpenRouterError, OpenRouterLLM
from chatchtg.llm.pricing import cost_usd


def _response(**choice_overrides):
    choice = {
        "message": {"role": "assistant", "content": "There are four doors."},
        "finish_reason": "stop",
    }
    choice.update(choice_overrides)
    return {
        "id": "gen-1",
        "model": "moonshotai/kimi-k3.1-pro-20261023",
        "provider": "Modal",
        "choices": [choice],
        "usage": {
            "prompt_tokens": 120,
            "completion_tokens": 900,
            "total_tokens": 1020,
            "completion_tokens_details": {"reasoning_tokens": 891},
        },
    }


class StubPost:
    """Stands in for the HTTP call: records the request body, returns a canned response."""

    def __init__(self, response=None):
        self.bodies = []
        self._response = response or _response()

    def __call__(self, body):
        self.bodies.append(body)
        return self._response


CONTEXT = Context(
    system="You answer questions about drawings.",
    parts=[ImagePart(media_type="image/png", data=PNG), TextPart(text="How many doors?")],
)


def test_request_carries_the_context_in_order():
    post = StubPost()
    OpenRouterLLM("qwen/qwen3.8-27b", post=post).complete(CONTEXT)

    (body,) = post.bodies
    assert body["model"] == "qwen/qwen3.8-27b"
    image_url = "data:image/png;base64," + base64.standard_b64encode(PNG).decode("ascii")
    assert body["messages"] == [
        {"role": "system", "content": "You answer questions about drawings."},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": "How many doors?"},
            ],
        },
    ]
    assert body["max_tokens"] > 0
    assert "temperature" not in body
    assert "stream" not in body


def test_request_with_a_provider_goes_only_to_that_provider():
    post = StubPost()
    OpenRouterLLM(provider="modal", post=post).complete(CONTEXT)

    assert post.bodies[0]["model"] == DEFAULT_MODEL == "moonshotai/kimi-k3.1-pro"
    assert post.bodies[0]["provider"] == {"only": ["modal"]}


def test_request_without_a_provider_lets_openrouter_choose():
    post = StubPost()
    OpenRouterLLM(post=post).complete(CONTEXT)

    assert "provider" not in post.bodies[0]


def test_completion_reports_text_requested_model_and_usage():
    completion = OpenRouterLLM(post=StubPost()).complete(CONTEXT)

    assert completion.text == "There are four doors."
    # The requested model, not the dated name in the response: it is the key of the price.
    assert completion.model == "moonshotai/kimi-k3.1-pro"
    assert completion.usage == {"input_tokens": 120, "output_tokens": 900}
    assert cost_usd(completion.model, completion.usage) == pytest.approx(0.00924)


def test_refusal_becomes_a_digit_free_answer():
    response = _response(
        message={"role": "assistant", "content": None, "refusal": "I cannot help."},
        finish_reason="content_filter",
    )
    completion = OpenRouterLLM(post=StubPost(response)).complete(CONTEXT)

    assert "declined" in completion.text
    assert not any(ch.isdigit() for ch in completion.text)


def test_budget_spent_on_reasoning_becomes_a_digit_free_answer():
    response = _response(message={"role": "assistant", "content": ""}, finish_reason="length")
    completion = OpenRouterLLM(post=StubPost(response)).complete(CONTEXT)

    assert completion.text.startswith("[no answer]")
    assert not any(ch.isdigit() for ch in completion.text)
    assert completion.usage == {"input_tokens": 120, "output_tokens": 900}


def test_answer_cut_off_by_the_limit_keeps_its_text():
    response = _response(
        message={"role": "assistant", "content": "There are"}, finish_reason="length"
    )
    completion = OpenRouterLLM(post=StubPost(response)).complete(CONTEXT)

    assert completion.text == "There are"


@pytest.mark.parametrize(
    "response",
    [
        # OpenRouter can send an error with status 200 and no choices.
        {"id": "gen-1", "error": {"code": 502, "message": "Provider returned error"}},
        _response(
            finish_reason="error",
            error={"code": 502, "message": "Provider returned error"},
        ),
    ],
)
def test_error_in_the_response_is_raised(response):
    with pytest.raises(OpenRouterError, match="Provider returned error"):
        OpenRouterLLM(post=StubPost(response)).complete(CONTEXT)


def test_build_llm_returns_openrouter_without_needing_a_key():
    default = build_llm("openrouter")
    assert isinstance(default, OpenRouterLLM)
    assert default.model == DEFAULT_MODEL
    assert default.provider is None

    other = build_llm("openrouter", "qwen/qwen3.8-27b", openrouter_provider="deepinfra")
    assert other.model == "qwen/qwen3.8-27b"
    assert other.provider == "deepinfra"


def test_unknown_provider_error_lists_openrouter():
    with pytest.raises(ValueError, match="openrouter"):
        build_llm("nope")


def test_missing_key_is_reported_before_any_request(monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", lambda *args, **kwargs: pytest.fail("sent"))

    with pytest.raises(OpenRouterError, match="CHATCHTG_OPENROUTER_API_KEY"):
        build_llm("openrouter").complete(CONTEXT)


def test_key_goes_to_openrouter_as_a_bearer_token(monkeypatch):
    sent = []

    def urlopen(request, timeout, context):
        sent.append(request)
        return io.BytesIO(json.dumps(_response()).encode())

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)

    completion = build_llm("openrouter", api_key="project-key").complete(CONTEXT)

    (request,) = sent
    assert request.full_url == "https://openrouter.ai/api/v1/chat/completions"
    assert request.get_method() == "POST"
    assert request.get_header("Authorization") == "Bearer project-key"
    assert json.loads(request.data)["model"] == DEFAULT_MODEL
    assert completion.text == "There are four doors."


def test_http_error_reports_the_message_and_not_the_key(monkeypatch):
    def urlopen(request, timeout, context):
        body = json.dumps({"error": {"code": 402, "message": "Insufficient credits"}}).encode()
        raise urllib.error.HTTPError(request.full_url, 402, "Payment Required", {}, io.BytesIO(body))

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)

    with pytest.raises(OpenRouterError, match="402.*Insufficient credits") as raised:
        build_llm("openrouter", api_key="project-key").complete(CONTEXT)
    assert "project-key" not in str(raised.value)


def test_settings_read_the_openrouter_key_and_provider():
    defaults = Settings.from_env({})
    assert defaults.openrouter_api_key is None
    assert defaults.openrouter_provider == "modal"

    settings = Settings.from_env(
        {"CHATCHTG_OPENROUTER_API_KEY": "or-key", "CHATCHTG_OPENROUTER_PROVIDER": "deepinfra"}
    )
    assert settings.openrouter_api_key == "or-key"
    assert settings.openrouter_provider == "deepinfra"
    assert "or-key" not in repr(settings)
    # An empty value removes the pin: OpenRouter then picks the endpoint.
    assert Settings.from_env({"CHATCHTG_OPENROUTER_PROVIDER": ""}).openrouter_provider is None


def test_wiring_gives_the_openrouter_key_and_the_modal_provider_to_the_adapter(library, tmp_path):
    from chatchtg.wiring import build_answer_service

    env = {
        "CHATCHTG_STATE_DIR": str(tmp_path / "var"),
        "CHATCHTG_LLM_PROVIDER": "openrouter",
        "CHATCHTG_ANTHROPIC_API_KEY": "anthropic-key",
        "CHATCHTG_OPENROUTER_API_KEY": "or-key",
    }
    llm = build_answer_service(Settings.from_env(env), library)._llm

    assert isinstance(llm, OpenRouterLLM)
    assert llm.provider == "modal"
    assert llm._api_key == "or-key"
