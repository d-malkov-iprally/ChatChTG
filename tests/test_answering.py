import pytest

from chatchtg.answering import AREA_PIPELINE_MODEL, AnswerService
from chatchtg.areas.pipeline import PlanAnalysis, PlanRoom
from chatchtg.context import areas
from chatchtg.context.areas import AreaComposer
from chatchtg.context.baseline import SYSTEM_PROMPT, BaselineComposer
from chatchtg.context.composer import ComposerInput
from chatchtg.context.registry import get_composer
from chatchtg.context.types import ComputedAnswer, Context, ImagePart, TextPart
from chatchtg.documents.library import DocumentNotFoundError
from chatchtg.evals.scoring import extract_number, score
from chatchtg.llm.client import Completion, build_llm
from chatchtg.llm.fake import FakeLLM
from chatchtg.schemas import AnswerRequest
from conftest import PNG, EvidenceComposer


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


def test_evidence_of_the_composer_reaches_the_response(library):
    document = library.add_upload(PNG, "plan.png")
    service = AnswerService(library, EvidenceComposer(), FakeLLM(reply="8.00 m²"))

    response = service.answer(AnswerRequest(document_id=document.id, message="Area of room 1.01?"))

    assert response.method == "printed"
    assert response.evidence[0]["point"] == [4, 5]


def test_a_plain_llm_answer_has_no_method_and_no_evidence(library, service):
    document = library.add_upload(PNG, "plan.png")

    response = service.answer(AnswerRequest(document_id=document.id, message="How many doors?"))

    assert response.method is None and response.evidence == []


class FixedComposer:
    """A composer that returns one context for every question."""

    name = "fixed"
    version = 1

    def __init__(self, context):
        self._context = context

    def compose(self, request):
        return self._context


class RecordingLLM:
    model = "recording"

    def __init__(self):
        self.calls = []

    def complete(self, context):
        self.calls.append(context)
        return Completion(text="The LLM says 21 m²", model=self.model, usage={"input_tokens": 5})


def computed_context(statement):
    return Context(
        system="Not sent.",
        parts=[TextPart(text="Result: 20.63 m²")],
        computed=ComputedAnswer(value=20.627, basis="measured", statement=statement),
    )


def test_a_computed_statement_is_the_answer_and_no_llm_is_called(library):
    document = library.add_upload(PNG, "plan.png")
    context = computed_context("Sum of the 2 rooms: 1.01, 1.02.\n\nAnswer: 20.63 m²")
    llm = RecordingLLM()
    service = AnswerService(library, FixedComposer(context), llm)

    trace = service.answer_traced(AnswerRequest(document_id=document.id, message="Area of apartment A?"))

    assert llm.calls == []
    assert trace.response.text.endswith("Answer: 20.63 m²")
    assert trace.response.model == AREA_PIPELINE_MODEL and trace.completion.usage == {}
    assert trace.response.composer == "fixed@1" and trace.context == context
    assert score("area", 20.6, trace.response.text) is True


def test_a_computed_answer_without_a_statement_goes_to_the_llm(library):
    document = library.add_upload(PNG, "plan.png")
    llm = RecordingLLM()
    service = AnswerService(library, FixedComposer(computed_context(None)), llm)

    trace = service.answer_traced(AnswerRequest(document_id=document.id, message="Area of apartment A?"))

    assert len(llm.calls) == 1 and trace.response.model == llm.model
    assert trace.response.text == "The LLM says 21 m²"


def test_the_area_composer_answers_in_code_end_to_end(library, monkeypatch):
    kitchen = PlanRoom(
        number="1.01", name="Küche", area=8.127, method="measured", printed_area=None,
        measured_area=8.127, apartment="A", outdoor=False, seed=(0.0, 0.0),
    )
    plan = PlanAnalysis(scale=100, scale_read=True, pixels_per_metre=59.055, rooms=[kitchen], missing_numbers=[])
    monkeypatch.setattr(areas, "analyse_bytes", lambda content: plan)
    document = library.add_upload(PNG, "plan.png")
    service = AnswerService(library, AreaComposer(), FakeLLM())

    response = service.answer(
        AnswerRequest(document_id=document.id, message="What is the net floor area of room 1.01 ('Küche') in m²?")
    )

    assert response.model == AREA_PIPELINE_MODEL and response.composer.startswith("areas@")
    assert extract_number(response.text) == 8.13


def test_the_area_composer_still_asks_the_llm_when_the_pipeline_cannot_answer(library, monkeypatch):
    plan = PlanAnalysis(scale=100, scale_read=True, pixels_per_metre=59.055, rooms=[], missing_numbers=[])
    monkeypatch.setattr(areas, "analyse_bytes", lambda content: plan)
    document = library.add_upload(PNG, "plan.png")
    service = AnswerService(library, AreaComposer(), FakeLLM(reply="About 8 m²"))

    response = service.answer(
        AnswerRequest(document_id=document.id, message="What is the net floor area of room 1.01 ('Küche') in m²?")
    )

    assert response.model == "fake" and response.text == "About 8 m²"
