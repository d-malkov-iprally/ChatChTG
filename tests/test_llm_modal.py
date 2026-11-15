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
from chatchtg.llm.modal import DEFAULT_MODEL, ModalError, ModalLLM
from chatchtg.llm.pricing import cost_usd

ENDPOINT = "kimi.us-west.modal.direct"
TOKEN = "wk-id.ws-secret"


def _response(**choice_overrides):
    choice = {
        "message": {"role": "assistant", "content": "There are four doors."},
        "finish_reason": "stop",
    }
    choice.update(choice_overrides)
    return {
        "id": "chatcmpl-1",
        "model": ENDPOINT,
        "choices": [choice],
        "usage": {"prompt_tokens": 120, "completion_tokens": 900, "total_tokens": 1020},
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


def _llm(response=None):
    post = StubPost(response)
    return ModalLLM(endpoint=ENDPOINT, post=post), post


def test_request_names_the_endpoint_and_carries_the_context_in_order():
    llm, post = _llm()
    llm.complete(CONTEXT)

    (body,) = post.bodies
    # Modal picks the endpoint by the model field; the name of the weights is not sent.
    assert body["model"] == ENDPOINT
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


def test_completion_reports_text_served_model_and_usage():
    llm, _ = _llm()
    completion = llm.complete(CONTEXT)

    assert completion.text == "There are four doors."
    # The name of the weights, not the endpoint: it is the key of the price.
    assert completion.model == DEFAULT_MODEL == "moonshotai/Kimi-K3.1-Pro"
    assert completion.usage == {"input_tokens": 120, "output_tokens": 900}
    assert cost_usd(completion.model, completion.usage) == pytest.approx(0.00924)


def test_budget_spent_on_reasoning_becomes_a_digit_free_answer():
    response = _response(message={"role": "assistant", "content": None}, finish_reason="length")
    llm, _ = _llm(response)
    completion = llm.complete(CONTEXT)

    assert completion.text.startswith("[no answer]")
    assert not any(ch.isdigit() for ch in completion.text)
    assert completion.usage == {"input_tokens": 120, "output_tokens": 900}


def test_answer_cut_off_by_the_limit_keeps_its_text():
    response = _response(
        message={"role": "assistant", "content": "There are"}, finish_reason="length"
    )
    llm, _ = _llm(response)

    assert llm.complete(CONTEXT).text == "There are"


@pytest.mark.parametrize(
    "response",
    [
        {"error": {"message": "The endpoint is not ready", "code": 503}},
        {"error": "The endpoint is not ready"},
    ],
)
def test_error_in_the_response_is_raised(response):
    llm, _ = _llm(response)
    with pytest.raises(ModalError, match="The endpoint is not ready"):
        llm.complete(CONTEXT)


def test_response_without_choices_is_raised():
    llm, _ = _llm({"id": "chatcmpl-1", "choices": []})
    with pytest.raises(ModalError, match="no completion"):
        llm.complete(CONTEXT)


def test_build_llm_returns_modal_without_needing_an_endpoint_or_a_token():
    default = build_llm("modal")
    assert isinstance(default, ModalLLM)
    assert default.model == DEFAULT_MODEL
    assert default.endpoint is None

    other = build_llm("modal", "moonshotai/Kimi-K3", api_key=TOKEN, modal_endpoint=ENDPOINT)
    assert other.model == "moonshotai/Kimi-K3"
    assert other.endpoint == ENDPOINT


def test_unknown_provider_error_lists_modal():
    with pytest.raises(ValueError, match="modal"):
        build_llm("nope")


@pytest.mark.parametrize(
    ("arguments", "setting"),
    [
        ({"api_key": TOKEN}, "CHATCHTG_MODAL_ENDPOINT"),
        ({"modal_endpoint": ENDPOINT}, "CHATCHTG_MODAL_PROXY_TOKEN"),
    ],
)
def test_missing_setting_is_reported_before_any_request(monkeypatch, arguments, setting):
    monkeypatch.setattr(urllib.request, "urlopen", lambda *args, **kwargs: pytest.fail("sent"))

    with pytest.raises(ModalError, match=setting):
        build_llm("modal", **arguments).complete(CONTEXT)


def test_proxy_token_goes_to_modal_as_a_bearer_token(monkeypatch):
    sent = []

    def urlopen(request, timeout, context):
        sent.append(request)
        return io.BytesIO(json.dumps(_response()).encode())

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)

    completion = build_llm("modal", api_key=TOKEN, modal_endpoint=ENDPOINT).complete(CONTEXT)

    (request,) = sent
    assert request.full_url == "https://inference.us-west.modal.direct/v1/chat/completions"
    assert request.get_method() == "POST"
    assert request.get_header("Authorization") == f"Bearer {TOKEN}"
    assert json.loads(request.data)["model"] == ENDPOINT
    assert completion.text == "There are four doors."


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (b'{"error": {"message": "Too many requests", "code": 429}}', "429: Too many requests"),
        (b'{"detail": "Too many requests"}', "429: Too many requests"),
        (b"modal-http: too many requests", "429: modal-http: too many requests"),
        (b"", "429: Too Many Requests"),
    ],
)
def test_http_error_reports_what_modal_said_and_not_the_token(monkeypatch, body, expected):
    def urlopen(request, timeout, context):
        raise urllib.error.HTTPError(
            request.full_url, 429, "Too Many Requests", {}, io.BytesIO(body)
        )

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)

    with pytest.raises(ModalError) as raised:
        build_llm("modal", api_key=TOKEN, modal_endpoint=ENDPOINT).complete(CONTEXT)
    assert expected in str(raised.value)
    assert TOKEN not in str(raised.value)


def test_response_that_is_not_json_is_raised(monkeypatch):
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda request, timeout, context: io.BytesIO(b"<html>")
    )

    with pytest.raises(ModalError, match="not a JSON object"):
        build_llm("modal", api_key=TOKEN, modal_endpoint=ENDPOINT).complete(CONTEXT)


def test_settings_read_the_modal_endpoint_and_token():
    defaults = Settings.from_env({})
    assert defaults.modal_endpoint is None
    assert defaults.modal_proxy_token is None

    settings = Settings.from_env(
        {"CHATCHTG_MODAL_ENDPOINT": ENDPOINT, "CHATCHTG_MODAL_PROXY_TOKEN": TOKEN}
    )
    assert settings.modal_endpoint == ENDPOINT
    assert settings.modal_proxy_token == TOKEN
    assert TOKEN not in repr(settings)


def test_wiring_gives_the_modal_endpoint_and_token_to_the_adapter(library, tmp_path):
    from chatchtg.wiring import build_answer_service

    env = {
        "CHATCHTG_STATE_DIR": str(tmp_path / "var"),
        "CHATCHTG_LLM_PROVIDER": "modal",
        "CHATCHTG_ANTHROPIC_API_KEY": "anthropic-key",
        "CHATCHTG_MODAL_ENDPOINT": ENDPOINT,
        "CHATCHTG_MODAL_PROXY_TOKEN": TOKEN,
    }
    llm = build_answer_service(Settings.from_env(env), library)._llm

    assert isinstance(llm, ModalLLM)
    assert llm.endpoint == ENDPOINT
    assert llm._proxy_token == TOKEN
