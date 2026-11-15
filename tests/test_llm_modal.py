import base64
import io
import json
import urllib.error
import urllib.request

import pytest
from conftest import PNG

from chatchtg.config import Settings
from PIL import Image

from chatchtg.context.types import Context, ImagePart, LocationRequest, TextPart
from chatchtg.llm.client import Location, build_llm
from chatchtg.llm.modal import (
    DEFAULT_MODEL,
    LOCATED_FORMAT_PROMPT,
    LOCATED_SCHEMA,
    UNREADABLE_TEXT,
    ModalError,
    ModalLLM,
    full_size_tokens,
    to_stored_pixels,
)
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


# --- Locations: the request, the reading of the reply, the coordinates ---

LOCATED = CONTEXT.model_copy(update={"locations": LocationRequest()})


def real_png(width, height):
    out = io.BytesIO()
    Image.new("L", (width, height), 255).save(out, format="PNG")
    return out.getvalue()


def _json_reply(payload, usage=None, **choice_overrides):
    text = payload if isinstance(payload, str) else json.dumps(payload)
    response = _response(message={"role": "assistant", "content": text}, **choice_overrides)
    if usage is not None:
        response["usage"] = usage
    return response


def test_plain_context_sends_the_request_of_today():
    llm, post = _llm()
    llm.complete(CONTEXT)
    assert set(post.bodies[0]) == {"model", "max_tokens", "messages"}
    assert llm.complete(CONTEXT).locations is None


def test_location_request_adds_the_schema_and_the_format_sentence():
    llm, post = _llm(_json_reply({"items": [], "answer": "There are 0 doors."}))
    completion = llm.complete(LOCATED)

    (body,) = post.bodies
    assert body["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "located_answer", "strict": True, "schema": LOCATED_SCHEMA},
    }
    assert body["messages"][0]["content"] == f"{CONTEXT.system} {LOCATED_FORMAT_PROMPT}"
    assert body["messages"][1] == post.bodies[0]["messages"][1]
    assert set(LOCATED_SCHEMA["required"]) == {"items", "answer"}
    assert (completion.text, completion.locations) == ("There are 0 doors.", [])


def test_reply_becomes_answer_text_and_locations_in_any_key_order():
    reply = {"answer": "There is 1 door.", "items": [{"label": "door", "x": 10, "y": 20.5}]}
    completion = _llm(_json_reply(reply))[0].complete(LOCATED)
    assert completion.text == "There is 1 door."
    assert completion.locations == [Location(label="door", point=(10, 20.5))]
    assert completion.usage == {"input_tokens": 120, "output_tokens": 900}


@pytest.mark.parametrize(
    "text",
    [
        '```json\n{"items": [{"label": "door", "x": 1, "y": 2}], "answer": "1 door"}\n```',
        'Here it is: {"items": [{"label": "door", "x": 1, "y": 2}], "answer": "1 door"} Done.',
    ],
)
def test_json_inside_other_text_is_still_read(text):
    completion = _llm(_json_reply(text))[0].complete(LOCATED)
    assert completion.text == "1 door"
    assert completion.locations == [Location(label="door", point=(1, 2))]


@pytest.mark.parametrize(
    "text",
    [
        "not json",
        "There are 4 doors at (10, 20).",
        '{"items": [{"label": "door", "x": 1',
        '{"items": []}',
        '{"items": [], "answer": 4}',
        '{"items": [], "answer": "  "}',
        "[1, 2]",
    ],
)
def test_unreadable_reply_gives_a_digit_free_text(text):
    completion = _llm(_json_reply(text, finish_reason="length"))[0].complete(LOCATED)
    assert completion.text == UNREADABLE_TEXT
    assert not any(ch.isdigit() for ch in completion.text)
    assert completion.locations is None
    assert completion.usage == {"input_tokens": 120, "output_tokens": 900}


def test_budget_spent_on_reasoning_with_a_location_request_says_so():
    response = _response(message={"role": "assistant", "content": None}, finish_reason="length")
    completion = _llm(response)[0].complete(LOCATED)
    assert completion.text.startswith("[no answer]") and "output limit" in completion.text
    assert completion.locations is None


def test_bad_items_do_not_lose_the_answer():
    whole = _json_reply({"items": "oops", "answer": "There are 2 doors."})
    completion = _llm(whole)[0].complete(LOCATED)
    assert (completion.text, completion.locations) == ("There are 2 doors.", None)

    one_bad = {
        "items": [
            {"label": "door", "x": "left", "y": 5},
            {"label": "door", "x": True, "y": 5},
            {"label": 3, "x": 1, "y": 5},
            "door",
            {"label": "door", "x": 7, "y": 8},
        ],
        "answer": "2 doors",
    }
    completion = _llm(_json_reply(one_bad))[0].complete(LOCATED)
    assert completion.locations == [Location(label="door", point=(7, 8))]


def test_model_sees_the_file_at_its_own_size_so_pixels_are_not_scaled():
    # 28-pixel patches of the whole file: the token counts that the endpoint reported.
    assert full_size_tokens((2480, 1753)) == 5607
    assert full_size_tokens((3507, 2480)) == 11214
    assert full_size_tokens((397, 479)) == 270
    assert to_stored_pixels(885, 607, (2480, 1753), 5607) == (885, 607)
    assert to_stored_pixels(885, 607, (2480, 1753), None) == (885, 607)   # no count reported
    assert to_stored_pixels(885, 607, None, 5607) == (885, 607)           # size not readable


def test_fewer_image_tokens_than_the_full_grid_give_no_position():
    # The endpoint made the image smaller; where the model's pixels then point is not known.
    assert to_stored_pixels(885, 607, (2480, 1753), 1500) is None


def _image_context(width, height):
    return Context(
        system="s",
        parts=[ImagePart(media_type="image/png", data=real_png(width, height)), TextPart(text="q")],
        locations=LocationRequest(),
    )


def test_positions_of_a_full_size_image_are_kept_as_pixels_of_the_file():
    usage = {"prompt_tokens": 6011, "completion_tokens": 763, "prompt_tokens_details": {"image_tokens": 5607}}
    reply = {"items": [{"label": "WC", "x": 885, "y": 607}], "answer": "1 toilet"}
    completion = _llm(_json_reply(reply, usage=usage))[0].complete(_image_context(2480, 1753))
    assert completion.locations == [Location(label="WC", point=(885, 607))]
    assert completion.usage == {"input_tokens": 6011, "output_tokens": 763}


def test_positions_are_left_out_when_the_endpoint_resized_the_image():
    usage = {"prompt_tokens": 1700, "completion_tokens": 763, "prompt_tokens_details": {"image_tokens": 1500}}
    reply = {"items": [{"label": "WC", "x": 500, "y": 300}], "answer": "1 toilet"}
    completion = _llm(_json_reply(reply, usage=usage))[0].complete(_image_context(2480, 1753))
    assert (completion.text, completion.locations) == ("1 toilet", None)
