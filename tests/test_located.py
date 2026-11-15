import io
import json

import pytest
from conftest import PNG, EvidenceComposer
from PIL import Image

from chatchtg.answering import AnswerService
from chatchtg.context.baseline import SYSTEM_PROMPT, BaselineComposer
from chatchtg.context.composer import ComposerInput, composer_id
from chatchtg.context.located import LOCATE_PROMPT, LocatedComposer, asks_for_count
from chatchtg.context.registry import get_composer
from chatchtg.context.types import Context, ImagePart, LocationRequest, TextPart
from chatchtg.evals.runner import run_eval
from chatchtg.evals.scoring import score
from chatchtg.evidence_check import CheckedEvidence, check_locations
from chatchtg.llm.client import Completion, Location
from chatchtg.llm.fake import FakeLLM
from chatchtg.schemas import AnswerRequest


def real_png(width: int, height: int) -> bytes:
    out = io.BytesIO()
    Image.new("L", (width, height), 255).save(out, format="PNG")
    return out.getvalue()


def test_image_part_reports_its_size_or_none():
    assert ImagePart(media_type="image/png", data=real_png(40, 30)).size() == (40, 30)
    assert ImagePart(media_type="image/png", data=PNG).size() is None


def test_existing_contexts_ask_for_no_locations(library):
    document = library.add_upload(PNG, "plan.png")
    context = BaselineComposer().compose(ComposerInput(document=document, content=PNG, message="Hi"))
    assert context.locations is None
    assert "locations" not in context.summary()


def test_summary_records_a_location_request():
    context = Context(system="s", parts=[TextPart(text="q")], locations=LocationRequest())
    assert context.summary()["locations"] == {"shape": "point"}


def test_fake_llm_returns_locations_only_when_asked():
    marks = [Location(label="door", point=(10, 20))]
    llm = FakeLLM(reply="There is 1 door.", locations=marks)
    plain = Context(system="s", parts=[TextPart(text="q")])
    asked = plain.model_copy(update={"locations": LocationRequest()})
    assert llm.complete(plain).locations is None
    assert llm.complete(asked).locations == marks
    assert FakeLLM().complete(asked).locations == []
    assert Completion(text="t", model="m").locations is None


@pytest.mark.parametrize("message", [
    "How many windows are shown on this floor plan?",
    "Kuinka monta ikkunaa pohjapiirustuksessa on?",
    "Wie viele Fenster sind auf diesem Grundriss eingezeichnet?",
    "Combien de baignoires sont dessinées sur ce plan ?",
    "Quanti locali con la denominazione «Stanza» ci sono in questo piano?",
    "  quante finestre?",
])
def test_count_questions_ask_for_locations(library, message):
    document = library.add_upload(PNG, "plan.png")
    context = LocatedComposer().compose(ComposerInput(document=document, content=PNG, message=message))
    assert context.locations == LocationRequest()
    assert context.system == f"{SYSTEM_PROMPT} {LOCATE_PROMPT}"
    assert context.parts == [ImagePart(media_type="image/png", data=PNG), TextPart(text=message)]
    assert context.method is None and context.evidence == []


@pytest.mark.parametrize("message", [
    "What is the net floor area of room 3 ('Bad') in m²?",
    "Wie gross ist die Nettofläche von Raum 4 («Zimmer») in m²?",
    "Describe this drawing.",
    "Quantity of windows?",
])
def test_other_questions_get_the_baseline_context(library, message):
    document = library.add_upload(PNG, "plan.png")
    request = ComposerInput(document=document, content=PNG, message=message)
    assert not asks_for_count(message)
    assert LocatedComposer().compose(request) == BaselineComposer().compose(request)


def test_located_is_listed_and_the_baseline_is_unchanged():
    assert composer_id(get_composer("located")) == "located@1"
    assert composer_id(get_composer("baseline")) == "baseline@1"


def marks(*points):
    return [Location(label="window", point=p) for p in points]


def test_consistent_locations_become_point_evidence():
    checked = check_locations(marks((10.4, 20.6), (300, 200)), (1000, 700), 2)
    assert checked.evidence == [
        {"text": "window 1", "kind": "llm", "point": [10, 21]},
        {"text": "window 2", "kind": "llm", "point": [300, 200]},
    ]
    assert checked.note is None


def test_count_that_differs_from_the_marks_is_reported_and_marks_are_kept():
    checked = check_locations(marks((10, 20), (300, 200)), (1000, 700), 3)
    assert len(checked.evidence) == 2
    assert checked.note == "The answer says 3, but 2 positions are marked."


def test_marks_outside_the_drawing_are_dropped():
    checked = check_locations(marks((1000, 5), (10, 20), (5, -1)), (1000, 700), 3)
    assert checked.evidence == [{"text": "window 2", "kind": "llm", "point": [10, 20]}]
    assert checked.note == "2 marked positions were outside the drawing and are not shown."


def test_near_duplicates_are_reported():
    checked = check_locations(marks((100, 100), (104, 103), (500, 500)), (1000, 700), 3)
    assert len(checked.evidence) == 3
    assert checked.note == "1 marks are very close to another mark and may show the same item."


def test_no_number_and_no_size_give_no_note():
    assert check_locations(marks((10, 20)), None, None).note is None      # "yksi keittiö"
    assert check_locations([], (1000, 700), 0) == CheckedEvidence(evidence=[], note=None)


def test_all_findings_are_joined_in_one_note():
    checked = check_locations(marks((10, 20), (12, 21), (5000, 5)), (1000, 700), 5)
    assert checked.note == (
        "The answer says 5, but 3 positions are marked. "
        "1 marked positions were outside the drawing and are not shown. "
        "1 marks are very close to another mark and may show the same item."
    )


def test_answer_carries_llm_evidence_and_the_rule_still_reads_the_count(library):
    document = library.add_upload(PNG, "plan.png")
    llm = FakeLLM(reply="There are 2 windows.", locations=marks((10, 20), (300, 200)))
    service = AnswerService(library, LocatedComposer(), llm)
    response = service.answer(AnswerRequest(document_id=document.id, message="How many windows?"))
    assert response.composer == "located@1"
    assert [item["point"] for item in response.evidence] == [[10, 20], [300, 200]]
    assert response.evidence_note is None and response.method == "llm-located"
    assert score("count", 2, response.text) is True


def test_an_answer_without_a_readable_number_keeps_its_evidence(library):
    document = library.add_upload(PNG, "plan.png")
    llm = FakeLLM(reply="Yksi keittiö.", locations=marks((10, 20)))
    service = AnswerService(library, LocatedComposer(), llm)
    response = service.answer(AnswerRequest(document_id=document.id, message="Kuinka monta keittiötä?"))
    assert len(response.evidence) == 1 and response.evidence_note is None


def test_a_count_of_zero_has_empty_evidence_and_no_note(library):
    document = library.add_upload(PNG, "plan.png")
    service = AnswerService(library, LocatedComposer(), FakeLLM(reply="There are 0 windows.", locations=[]))
    response = service.answer(AnswerRequest(document_id=document.id, message="How many windows?"))
    assert response.method == "llm-located" and response.evidence == [] and response.evidence_note is None


def test_marks_outside_a_real_image_are_dropped_by_the_service(library):
    document = library.add_upload(real_png(40, 30), "plan.png")
    llm = FakeLLM(reply="2 windows", locations=marks((10, 20), (41, 5)))
    service = AnswerService(library, LocatedComposer(), llm)
    response = service.answer(AnswerRequest(document_id=document.id, message="How many windows?"))
    assert [item["text"] for item in response.evidence] == ["window 1"]
    assert "outside the drawing" in response.evidence_note


def test_baseline_answers_carry_no_evidence(library, service):
    document = library.add_upload(PNG, "plan.png")
    response = service.answer(AnswerRequest(document_id=document.id, message="How many windows?"))
    assert response.method is None and response.evidence == [] and response.evidence_note is None


def test_the_evidence_of_the_context_wins_over_the_locations_of_the_completion(library, caplog):
    class Both(EvidenceComposer):
        def compose(self, request):
            return super().compose(request).model_copy(update={"locations": LocationRequest()})

    document = library.add_upload(PNG, "plan.png")
    service = AnswerService(library, Both(), FakeLLM(reply="8.00 m²", locations=marks((1, 2))))
    response = service.answer(AnswerRequest(document_id=document.id, message="Area?"))
    assert response.method == "printed" and response.evidence_note is None
    assert [item["text"] for item in response.evidence] == ["1.01"]
    assert "not used" in caplog.text


@pytest.fixture
def one_question(data_root):
    dataset_dir = data_root / "preview"
    (dataset_dir / "data").mkdir(parents=True)
    (dataset_dir / "data" / "a.png").write_bytes(PNG)
    question = {"path": "data/a.png", "question": "How many windows?", "answer": 3, "unit": "count", "type": "count"}
    (dataset_dir / "questions.jsonl").write_text(json.dumps(question) + "\n")
    return dataset_dir


def test_eval_row_has_the_llm_evidence_and_the_note_only_with_the_option(one_question, library, tmp_path):
    service = AnswerService(library, LocatedComposer(), FakeLLM(reply="3 windows", locations=marks((1, 2))))

    run_eval(one_question, library, service, tmp_path / "with.jsonl", evidence=True)
    row = json.loads((tmp_path / "with.jsonl").read_text())
    assert row["method"] == "llm-located"
    assert row["evidence"] == [{"text": "window 1", "kind": "llm", "point": [1, 2]}]
    assert row["evidence_note"] == "The answer says 3, but 1 positions are marked."
    assert row["context"]["locations"] == {"shape": "point"}
    assert row["correct"] is True

    run_eval(one_question, library, service, tmp_path / "without.jsonl")
    row = json.loads((tmp_path / "without.jsonl").read_text())
    assert not {"method", "evidence", "evidence_note"} & set(row)
    assert row["context"]["locations"] == {"shape": "point"}
