import pytest

from chatchtg.answering import AnswerService
from chatchtg.context.baseline import SYSTEM_PROMPT, BaselineComposer
from chatchtg.context.composer import ComposerInput
from chatchtg.context.registry import get_composer
from chatchtg.context.types import ImagePart, TextPart
from chatchtg.documents.library import DocumentNotFoundError
from chatchtg.llm.client import build_llm
from chatchtg.llm.fake import FakeLLM
from chatchtg.schemas import AnswerRequest
from conftest import PNG


def test_baseline_composer_needs_no_storage_or_llm(library):
    document = library.add_upload(PNG, "plan.png")

    context = BaselineComposer().compose(
        ComposerInput(document=document, content=PNG, message="How many doors?")
    )

    assert context.system == SYSTEM_PROMPT
    assert context.parts == [
        ImagePart(media_type="image/png", data=PNG),
        TextPart(text="How many doors?"),
    ]


def test_context_summary_is_json_safe_and_omits_image_bytes(library):
    document = library.add_upload(PNG, "plan.png")
    context = BaselineComposer().compose(ComposerInput(document=document, content=PNG, message="Hi"))

    image, text = context.summary()["parts"]

    assert image == {
        "type": "image",
        "media_type": "image/png",
        "sha256": document.sha256,
        "size_bytes": len(PNG),
    }
    assert text == {"type": "text", "text": "Hi"}


def test_answer_names_the_composer_and_the_model(library, service):
    document = library.add_upload(PNG, "plan.png")

    response = service.answer(AnswerRequest(document_id=document.id, message="How many doors?"))

    assert "How many doors?" in response.text
    assert response.composer == "baseline@1"
    assert response.model == "fake"


def test_traced_answer_exposes_the_context_sent_to_the_llm(library):
    document = library.add_upload(PNG, "plan.png")
    service = AnswerService(library, BaselineComposer(), FakeLLM(reply="There are 4 doors."))

    trace = service.answer_traced(AnswerRequest(document_id=document.id, message="How many doors?"))

    assert trace.response.text == "There are 4 doors."
    assert trace.context.parts[0].data == PNG


def test_unknown_document_is_reported(service):
    with pytest.raises(DocumentNotFoundError):
        service.answer(AnswerRequest(document_id="missing", message="Hi"))


def test_unknown_composer_and_provider_are_rejected():
    assert get_composer("baseline").name == "baseline"
    with pytest.raises(ValueError, match="Unknown composer"):
        get_composer("nope")
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        build_llm("nope")
