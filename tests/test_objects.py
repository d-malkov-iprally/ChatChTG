import hashlib
import json
from pathlib import Path

import pytest

from chatchtg.answering import OBJECT_COUNTER_MODEL, AnswerService
from chatchtg.config import Settings
from chatchtg.context.baseline import BaselineComposer
from chatchtg.evals.scoring import extract_number
from chatchtg.extractions.models import Extraction, ImageSize, Item
from chatchtg.extractions.store import LocalExtractionStore
from chatchtg.llm.fake import FakeLLM
from chatchtg.llm.pricing import cost_usd
from chatchtg.objects import tune
from chatchtg.objects.counter import (
    PRODUCER,
    PRODUCER_VERSION,
    DetectorObjectCounter,
    ObjectFinding,
    count,
)
from chatchtg.objects.importer import import_detections, to_extraction
from chatchtg.objects.questions import ObjectQuestion, classify
from chatchtg.rooms.questions import classify as classify_room
from chatchtg.rooms.router import Finding
from chatchtg.rooms.questions import RoomQuestion
from chatchtg.schemas import AnswerRequest
from chatchtg.wiring import build_answer_service
from conftest import PNG

# Every distinct count question of the benchmark, and some of its area questions.
QUESTIONS = [
    json.loads(line)
    for line in (Path(__file__).parent / "fixtures" / "benchmark_questions.jsonl").read_text().splitlines()
]
EXPECTED = {
    "How many windows are shown on this floor plan?": "window",
    "Kuinka monta ikkunaa pohjapiirustuksessa on?": "window",
    "How many doors are shown on this floor plan? Count doorways fitted with a door; open "
    "doorways without a door leaf do not count.": "door",
    "Kuinka monta ovea pohjapiirustuksessa on? Laske oviaukot, joissa on ovi; pelkkiä "
    "oviaukkoja ilman ovilehteä ei lasketa.": "door",
    "How many toilet bowls (WC seats) are drawn on this floor plan?": "toilet",
    "Kuinka monta WC-istuinta pohjapiirustukseen on piirretty?": "toilet",
    "How many sinks and wash basins (kitchen and bathroom combined; a double-bowl sink counts "
    "as one) are drawn on this floor plan?": "sink",
    "Kuinka monta pesu- ja tiskiallasta (keittiö ja kylpytilat yhteensä; kaksoisallas "
    "lasketaan yhdeksi) pohjapiirustukseen on piirretty?": "sink",
    "How many showers and bathtubs are drawn on this floor plan in total?": "bath_shower",
    "Kuinka monta suihkua ja kylpyammetta pohjapiirustukseen on piirretty yhteensä?": "bath_shower",
}


def test_classifier_claims_exactly_the_ten_finnish_object_questions_of_the_benchmark():
    claimed = {}
    for row in QUESTIONS:
        question = classify(row["question"])
        if question is not None:
            claimed[row["question"]] = question.kind
            assert row["source"] == "cubicasa5k" and row["type"] == "count"
    assert claimed == EXPECTED


def test_classifier_and_room_classifier_never_claim_the_same_question():
    for row in QUESTIONS:
        assert classify(row["question"]) is None or classify_room(row["question"]) is None


def test_classifier_ignores_case_and_spacing_and_leaves_swiss_wording_alone():
    assert classify("  how many WINDOWS are  shown here?") == ObjectQuestion("window")
    assert classify("How many windows are drawn on this floor plan?") is None
    assert classify("How many showers are drawn on this floor plan?") is None
    assert ObjectQuestion("bath_shower").classes == ("shower", "bathtub")


def _extraction(document_id: str = "a" * 64, items=()) -> Extraction:
    return Extraction(
        document_id=document_id,
        producer=PRODUCER,
        version=PRODUCER_VERSION,
        image=ImageSize(width=100, height=100),
        attributes={"cubicasa_split": "test"},
        items=list(items),
    )


def _item(item_id: str, label: str, score: float) -> Item:
    return Item(id=item_id, kind="object", label=label, box=(1.2, 2, 10.6, 20), score=score)


def test_count_keeps_the_detections_above_the_minimum_of_their_class():
    extraction = _extraction(
        items=[
            _item("shower-1", "shower", 0.9),
            _item("shower-2", "shower", 0.2),
            _item("bathtub-1", "bathtub", 0.6),
            _item("window-1", "window", 0.99),
        ]
    )
    thresholds = {"shower": 0.5, "bathtub": 0.5, "window": 0.5}

    finding = count(extraction, ObjectQuestion("bath_shower"), thresholds)

    assert finding.count == 2
    assert finding.method == "detector"
    assert finding.evidence == [
        {"text": "shower 0.90", "box": [1, 2, 11, 20], "ref": f"{PRODUCER}@1#shower-1"},
        {"text": "bathtub 0.60", "box": [1, 2, 11, 20], "ref": f"{PRODUCER}@1#bathtub-1"},
    ]


def test_counter_answers_only_object_questions_on_drawings_with_detections(tmp_path):
    store = LocalExtractionStore(tmp_path / "metadata")
    store.put(_extraction("b" * 64, [_item("door-1", "door", 0.8)]))
    counter = DetectorObjectCounter(store, thresholds={"door": 0.5})

    assert counter.solve("b" * 64, "How many doors are shown on this floor plan?").count == 1
    assert counter.solve("c" * 64, "How many doors are shown on this floor plan?") is None
    assert counter.solve("b" * 64, "How many saunas are on this floor plan?") is None


def test_importer_stores_a_prediction_under_the_hash_of_the_benchmark_image(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    (images / "colorful-1.png").write_bytes(PNG)
    (images / "colorful-2.png").write_bytes(PNG + b"changed")
    row = {
        "image": "colorful-1.png",
        "plan": "/colorful/1/",
        "split": "valid",
        "in_dataset": True,
        "width": 50,
        "height": 40,
        "sha256": hashlib.sha256(PNG).hexdigest(),
        "detections": [
            {"label": "window", "score": 0.4, "box": [0, 0, 5, 5]},
            {"label": "sauna bench", "score": 0.7, "box": [1, 1, 6, 6]},
            {"label": "window", "score": 0.9, "box": [2, 2, 7, 7]},
        ],
    }
    other = {**row, "image": "colorful-2.png", "plan": "/colorful/2/"}
    detections = tmp_path / "detections.jsonl"
    detections.write_text(json.dumps(row) + "\n" + json.dumps(other) + "\n")
    store = LocalExtractionStore(tmp_path / "metadata")

    outcome = import_detections(detections, images, store, {"run": "x"})

    assert outcome == {"in the benchmark": 1, "image differs": 1}
    stored = store.get(row["sha256"], PRODUCER)
    assert stored.source == "data/cubicasa5k/colorful-1.png"
    assert stored.attributes["cubicasa_split"] == "valid"
    assert stored.attributes["in_training_data"] is False
    assert [(i.id, i.score) for i in stored.items] == [
        ("sauna-bench-1", 0.7),
        ("window-1", 0.9),
        ("window-2", 0.4),
    ]
    assert to_extraction({**row, "split": "train"}).attributes["in_training_data"] is True


def test_tune_picks_the_minimum_that_gives_the_most_exact_counts():
    answers = {"a.png": {"window": 2, "bath_shower": 1}, "b.png": {"window": 1}}
    scores = {
        "a.png": {"window": [0.9, 0.7, 0.3], "shower": [0.8], "bathtub": [0.2]},
        "b.png": {"window": [0.95, 0.5]},
    }

    result = tune.tune(answers, scores)

    # 0.51 to 0.70 give both counts right; the middle of that range is chosen.
    assert result["window"]["minimums"] == {"window": 0.61}
    assert result["window"]["exact"] == 2
    assert result["bath_shower"]["exact"] == 1


class FixedObjectCounter:
    def __init__(self, finding):
        self.finding = finding

    def solve(self, document_id, message):
        return self.finding


class FixedRoomCounter:
    def __init__(self):
        self.asked = False

    def solve(self, content, message):
        self.asked = True
        return Finding(RoomQuestion("bedroom"), 3, "labels", True)


def test_object_count_is_answered_from_detections_without_the_llm(library):
    document = library.add_upload(PNG, "plan.png")
    evidence = [{"text": "window 0.90", "box": [1, 2, 3, 4], "ref": f"{PRODUCER}@1#window-1"}]
    finding = ObjectFinding(ObjectQuestion("window"), 1, "detector", evidence)
    rooms = FixedRoomCounter()
    service = AnswerService(
        library, BaselineComposer(), FakeLLM(reply="7"), rooms, object_counter=FixedObjectCounter(finding)
    )

    response = service.answer(AnswerRequest(document_id=document.id, message="How many windows?"))

    assert response.model == OBJECT_COUNTER_MODEL
    assert response.method == "detector"
    assert response.evidence == evidence
    assert extract_number(response.text) == 1
    assert not rooms.asked


def test_without_detections_the_room_counter_and_the_llm_answer_as_before(library):
    document = library.add_upload(PNG, "plan.png")
    request = AnswerRequest(document_id=document.id, message="How many bedrooms?")
    service = AnswerService(
        library,
        BaselineComposer(),
        FakeLLM(reply="7"),
        FixedRoomCounter(),
        object_counter=FixedObjectCounter(None),
    )
    llm_only = AnswerService(
        library, BaselineComposer(), FakeLLM(reply="7"), object_counter=FixedObjectCounter(None)
    )

    assert service.answer(request).method == "labels"
    assert llm_only.answer(request).text == "7"
    assert llm_only.answer(request).method is None


def test_the_object_counter_is_off_by_default_and_reads_the_extraction_store(tmp_path, library):
    document = library.add_upload(PNG, "plan.png")
    settings = Settings.from_env({"CHATCHTG_STATE_DIR": str(tmp_path / "var")})
    request = AnswerRequest(document_id=document.id, message="How many windows are shown on this floor plan?")
    LocalExtractionStore(tmp_path / "var" / "metadata").put(
        _extraction(document.id, [_item("window-1", "window", 0.99), _item("window-2", "window", 0.97)])
    )

    assert settings.object_counter == "off"
    assert build_answer_service(settings, library).answer(request).model == "fake"

    on = Settings.from_env(
        {"CHATCHTG_STATE_DIR": str(tmp_path / "var"), "CHATCHTG_OBJECT_COUNTER": "detections"}
    )
    response = build_answer_service(on, library).answer(request)
    assert response.model == OBJECT_COUNTER_MODEL
    assert extract_number(response.text) == 2
    assert len(response.evidence) == 2


def test_an_unknown_object_counter_mode_is_refused(library):
    settings = Settings.from_env({"CHATCHTG_OBJECT_COUNTER": "yes"})
    with pytest.raises(ValueError, match="object counter"):
        build_answer_service(settings, library)


def test_count_uses_the_minimum_scores_of_the_extractions_version(monkeypatch):
    from chatchtg.objects import counter as counter_module

    monkeypatch.setitem(counter_module.THRESHOLDS_BY_VERSION, 7, {"door": 0.95})
    door = _item("door-1", "door", 0.9)
    old = _extraction(items=[door])
    new = old.model_copy(update={"version": 7})

    assert count(old, ObjectQuestion("door")).count == 1
    assert count(new, ObjectQuestion("door")).count == 0


def test_llm_comparison_counts_each_answer_once_per_file(tmp_path, capsys):
    from chatchtg.objects import report

    store = LocalExtractionStore(tmp_path / "metadata")
    store.put(_extraction("d" * 64, [_item("window-1", "window", 0.99)]))
    row = {
        "path": "data/cubicasa5k/a.png",
        "document_id": "d" * 64,
        "question": "How many windows are shown on this floor plan?",
        "expected": 1,
        "correct": False,
        "model": "claude-opus-5-5",
    }
    other = {**row, "model": "fake"}
    run = tmp_path / "run.jsonl"
    run.write_text("\n".join(json.dumps(r) for r in (row, row, other)) + "\n")

    report.compare_llm([run], store)

    out = capsys.readouterr().out
    assert "1 answers on 1 plans" in out
    assert "window       test      1     0        1     0        0" in out


def test_an_answer_of_the_object_counter_has_a_price_of_zero():
    # Without a price, an eval with --max-cost stops at the first answer of the counter.
    assert cost_usd(OBJECT_COUNTER_MODEL, {}) == 0


class BrokenStore:
    def get(self, document_id, producer, version=None):
        raise ValueError("not an extraction")


def test_detections_that_cannot_be_read_leave_the_question_to_the_llm():
    counter = DetectorObjectCounter(BrokenStore())

    assert counter.solve("0" * 64, "How many windows are shown in this floor plan?") is None
