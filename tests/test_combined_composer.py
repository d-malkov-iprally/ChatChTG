"""The combined composer: area questions to the area composer, the rest to the room conventions."""

import json

import pytest

from chatchtg.answering import AnswerService
from chatchtg.areas.pipeline import PlanAnalysis, PlanRoom
from chatchtg.config import Settings
from chatchtg.context import areas
from chatchtg.context.areas import AREA_SYSTEM_PROMPT, AreaComposer
from chatchtg.context.baseline import BaselineComposer
from chatchtg.context.combined import CombinedComposer
from chatchtg.context.composer import ComposerInput, composer_id, context_composer
from chatchtg.context.registry import get_composer
from chatchtg.context.room_conventions import RoomConventionsComposer
from chatchtg.evals.runner import run_eval
from chatchtg.llm.fake import FakeLLM
from chatchtg.schemas import AnswerRequest
from conftest import PNG

AREA = "What is the net floor area of room 1.01 ('Küche') in m²?"
AREA_DE = "Wie gross ist die Nettowohnfläche von Wohnung A in m² (Summe aller Räume ohne Balkone)?"
BEDROOMS = "How many bedrooms (makuuhuone, usually marked MH) are on this floor plan?"
SAUNAS_FI = "Kuinka monta saunaa pohjapiirustuksessa on?"
OTHERS = (
    "How many windows are shown on this floor plan?",
    "How many rooms labelled 'Zimmer' are there on this floor?",
    "How many apartments (residential units) are shown on this floor plan?",
    "Is there a lift?",
)


@pytest.fixture(autouse=True)
def plan(monkeypatch):
    """The area pipeline returns a fixed analysis: no OCR in these tests."""
    rooms = [
        PlanRoom(
            number="1.01", name="Küche", area=8.127, method="measured", printed_area=None,
            measured_area=8.127, apartment="A", outdoor=False, seed=(0.0, 0.0),
        )
    ]
    result = PlanAnalysis(
        scale=100, scale_read=True, pixels_per_metre=59.055, rooms=rooms, missing_numbers=[]
    )
    monkeypatch.setattr(areas, "analyse_bytes", lambda content: result)


@pytest.fixture
def combined():
    return CombinedComposer(areas=AreaComposer(), room_conventions=RoomConventionsComposer())


@pytest.fixture
def request_for(library):
    document = library.add_upload(PNG, "plan.png")
    return lambda message: ComposerInput(document=document, content=PNG, message=message)


@pytest.mark.parametrize("message", [AREA, AREA_DE])
def test_an_area_question_gets_the_area_composers_context(combined, request_for, message):
    request = request_for(message)

    context = combined.compose(request)

    assert context == AreaComposer().compose(request)
    assert combined.delegate(request) is combined.parts[0]
    if message == AREA:
        assert context.system == AREA_SYSTEM_PROMPT


@pytest.mark.parametrize("message", [BEDROOMS, SAUNAS_FI])
def test_a_finnish_room_count_gets_the_conventions(combined, request_for, message):
    request = request_for(message)

    context = combined.compose(request)

    assert context == RoomConventionsComposer().compose(request)
    assert "Counting conventions" in context.parts[1].text
    assert combined.delegate(request) is combined.parts[1]


@pytest.mark.parametrize("message", OTHERS)
def test_any_other_question_gets_the_baseline_context(combined, request_for, message):
    request = request_for(message)

    assert combined.compose(request) == BaselineComposer().compose(request)


def test_the_id_names_the_choice_and_the_versions_of_its_parts(combined):
    expected = f"combined@1+{composer_id(AreaComposer())}+{composer_id(RoomConventionsComposer())}"
    assert composer_id(combined) == expected
    assert composer_id(BaselineComposer()) == "baseline@1"


def test_context_composer_names_the_part_that_built_the_context(combined, request_for):
    assert context_composer(combined, request_for(AREA)).name == "areas"
    assert context_composer(combined, request_for(BEDROOMS)).name == "room-conventions"
    assert context_composer(combined, request_for(OTHERS[0])).name == "room-conventions"
    assert context_composer(BaselineComposer(), request_for(AREA)) is None


def test_the_combined_composer_is_registered_and_baseline_stays_the_default():
    combined = get_composer("combined")
    assert combined.parts == (get_composer("areas"), get_composer("room-conventions"))
    assert Settings.from_env().composer == "baseline"


def test_the_response_records_the_composer_and_the_part(library):
    document = library.add_upload(PNG, "plan.png")
    service = AnswerService(library, get_composer("combined"), FakeLLM())

    area = service.answer(AnswerRequest(document_id=document.id, message=AREA))
    rooms = service.answer(AnswerRequest(document_id=document.id, message=BEDROOMS))

    assert area.composer == composer_id(get_composer("combined"))
    assert area.context_composer == composer_id(get_composer("areas"))
    assert rooms.context_composer == composer_id(get_composer("room-conventions"))
    baseline = AnswerService(library, BaselineComposer(), FakeLLM())
    assert baseline.answer(AnswerRequest(document_id=document.id, message=AREA)).context_composer is None


def test_an_eval_row_records_the_part(data_root, library, tmp_path):
    dataset = data_root / "set"
    (dataset / "data").mkdir(parents=True)
    (dataset / "data" / "a.png").write_bytes(PNG)
    questions = [
        {"path": "data/a.png", "question": AREA, "answer": 8.13, "unit": "m2", "type": "area"},
        {"path": "data/a.png", "question": BEDROOMS, "answer": 2, "unit": "count", "type": "count"},
    ]
    (dataset / "questions.jsonl").write_text("\n".join(json.dumps(q) for q in questions) + "\n")
    service = AnswerService(library, get_composer("combined"), FakeLLM())
    out_path = tmp_path / "result.jsonl"

    run_eval(dataset, library, service, out_path)

    rows = [json.loads(line) for line in out_path.read_text().splitlines()]
    assert [row["context_composer"] for row in rows] == [
        composer_id(get_composer("areas")),
        composer_id(get_composer("room-conventions")),
    ]
    assert rows[0]["composer"].startswith("combined@1+")
