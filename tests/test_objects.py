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
    CUBICASA,
    DEFAULT_SKIP,
    PRODUCER,
    PRODUCER_VERSION,
    SWISS,
    DetectorObjectCounter,
    ObjectFinding,
    count,
    parse_skip,
)
from chatchtg.objects.importer import (
    import_detections,
    import_swiss_detections,
    run_params,
    to_extraction,
)
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
SWISS_EXPECTED = {
    "How many doors (interior doors and entrance doors together) are drawn on this floor plan?": "door",
    "Wie viele Türen (Innentüren und Eingangstüren zusammen) sind auf diesem Grundriss eingezeichnet?": "door",
    "Combien de portes (portes intérieures et portes d'entrée confondues) sont dessinées sur ce plan ?": "door",
    "Quante porte (porte interne e porte d'ingresso insieme) sono disegnate in questa pianta?": "door",
    "How many windows are drawn on this floor plan?": "window",
    "Wie viele Fenster sind auf diesem Grundriss eingezeichnet?": "window",
    "Combien de fenêtres sont dessinées sur ce plan ?": "window",
    "Quante finestre sono disegnate in questa pianta?": "window",
    "How many toilets (WC) are drawn on this floor plan?": "toilet",
    "Wie viele WCs sind auf diesem Grundriss eingezeichnet?": "toilet",
    "Combien de WC sont dessinés sur ce plan ?": "toilet",
    "Quanti WC sono disegnati in questa pianta?": "toilet",
    "How many washbasins are drawn on this floor plan?": "washbasin",
    "Wie viele Lavabos sind auf diesem Grundriss eingezeichnet?": "washbasin",
    "Combien de lavabos sont dessinés sur ce plan ?": "washbasin",
    "Quanti lavabi sono disegnati in questa pianta?": "washbasin",
    "How many bathtubs are drawn on this floor plan?": "bathtub",
    "Wie viele Badewannen sind auf diesem Grundriss eingezeichnet?": "bathtub",
    "Combien de baignoires sont dessinées sur ce plan ?": "bathtub",
    "Quante vasche da bagno sono disegnate in questa pianta?": "bathtub",
    "How many showers are drawn on this floor plan?": "shower",
    "Wie viele Duschen sind auf diesem Grundriss eingezeichnet?": "shower",
    "Combien de douches sont dessinées sur ce plan ?": "shower",
    "Quante docce sono disegnate in questa pianta?": "shower",
}


def test_classifier_claims_exactly_the_object_questions_of_the_benchmark():
    # The fixture holds every distinct count question of the benchmark: the room-label and
    # apartment questions are among them, and must not be claimed.
    claimed = {}
    for row in QUESTIONS:
        question = classify(row["question"])
        if question is not None:
            claimed[(row["source"], row["question"])] = question.kind
            assert row["type"] == "count"
    assert claimed == {
        **{("cubicasa5k", q): kind for q, kind in EXPECTED.items()},
        **{("swiss_dwellings", q): kind for q, kind in SWISS_EXPECTED.items()},
    }


def test_classifier_and_room_classifier_never_claim_the_same_question():
    for row in QUESTIONS:
        assert classify(row["question"]) is None or classify_room(row["question"]) is None


def test_classifier_ignores_case_and_spacing():
    assert classify("  how many WINDOWS are  shown here?") == ObjectQuestion("window")
    assert classify("WIE VIELE  Duschen sind hier?") == ObjectQuestion("shower")
    assert classify("How many rooms labelled 'Bad' are on this floor plan?") is None
    assert CUBICASA.classes["bath_shower"] == ("shower", "bathtub")
    assert SWISS.classes["shower"] == ("shower",)


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
    counter = DetectorObjectCounter(store)
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

    report.compare_llm([run], store, counter)

    out = capsys.readouterr().out
    assert "1 answers on 1 plans" in out
    assert "cubicasa  window       test      1     0        1     0        0" in out


def test_an_answer_of_the_object_counter_has_a_price_of_zero():
    # Without a price, an eval with --max-cost stops at the first answer of the counter.
    assert cost_usd(OBJECT_COUNTER_MODEL, {}) == 0


class BrokenStore:
    def get(self, document_id, producer, version=None):
        raise ValueError("not an extraction")


def test_detections_that_cannot_be_read_leave_the_question_to_the_llm():
    counter = DetectorObjectCounter(BrokenStore())

    assert counter.solve("0" * 64, "How many windows are shown in this floor plan?") is None


def test_the_importer_refuses_detections_of_another_run():
    # Stored as version 1 they would replace the detections that the app reads.
    with pytest.raises(ValueError, match="large-1536-ft"):
        run_params({"run_name": "large-1536-ft"}, 1)
    with pytest.raises(ValueError, match="No run is named"):
        run_params(None, 99)


def test_the_importer_records_the_run_of_its_version():
    assert run_params({"run_name": "large-1536-ft", "min_score": 0.05}, 2) == {
        "run_name": "large-1536-ft", "min_score": 0.05, "checkpoint": "checkpoint_best_total.pth",
    }
    assert run_params(None, 1)["run_name"] == "large-1024"


# The Swiss Dwellings detector: a second profile of the same counter.

SWISS_DOORS = "How many doors (interior doors and entrance doors together) are drawn on this floor plan?"


def _swiss(document_id: str = "e" * 64, items=(), split: str = "test") -> Extraction:
    return Extraction(
        document_id=document_id,
        producer=SWISS.producer,
        version=SWISS.version,
        image=ImageSize(width=100, height=100),
        source="data/swiss_dwellings/site1_bldg1_floor1.png",
        attributes={"split": split, "in_training_data": split == "train"},
        items=list(items),
    )


def test_a_swiss_question_is_counted_from_the_swiss_detections_with_their_minimums(tmp_path):
    store = LocalExtractionStore(tmp_path / "metadata")
    # The Swiss minimum for washbasins is 0.70, for doors 0.55.
    store.put(
        _swiss(
            items=[
                _item("washbasin-1", "washbasin", 0.9),
                _item("washbasin-2", "washbasin", 0.6),
                _item("door-1", "door", 0.6),
                _item("washing-machine-1", "washing machine", 0.99),
            ]
        )
    )
    counter = DetectorObjectCounter(store, skip=())

    finding = counter.solve("e" * 64, "Wie viele Lavabos sind auf diesem Grundriss eingezeichnet?")
    assert finding.count == 1
    assert finding.method == "detector"
    assert finding.evidence[0]["ref"] == f"{SWISS.producer}@1#washbasin-1"
    assert counter.solve("e" * 64, SWISS_DOORS).count == 1
    # A Finnish template whose kind the Swiss detector does not answer goes to the LLM.
    assert counter.solve("e" * 64, "How many sinks and wash basins are drawn on this floor plan?") is None
    # No stored detections: the LLM answers.
    assert counter.solve("f" * 64, SWISS_DOORS) is None


def test_skipped_kinds_are_left_to_the_llm(tmp_path):
    store = LocalExtractionStore(tmp_path / "metadata")
    store.put(_swiss(items=[_item("door-1", "door", 0.9), _item("toilet-1", "toilet", 0.9)]))
    counter = DetectorObjectCounter(store, skip=parse_skip("swiss:door, swiss:window"))

    assert counter.solve("e" * 64, SWISS_DOORS) is None
    assert counter.solve("e" * 64, "Quanti WC sono disegnati in questa pianta?").count == 1


def test_the_skip_setting_names_known_profiles_and_kinds():
    assert parse_skip("") == frozenset()
    assert parse_skip("cubicasa:window,swiss:shower") == {("cubicasa", "window"), ("swiss", "shower")}
    with pytest.raises(ValueError, match="Unknown object counter kind"):
        parse_skip("swiss:sink")
    with pytest.raises(ValueError, match="Unknown object counter kind"):
        parse_skip("door")


def test_the_skip_setting_reaches_the_app(tmp_path, library):
    document = library.add_upload(PNG, "plan.png")
    LocalExtractionStore(tmp_path / "var" / "metadata").put(_swiss(document.id, [_item("door-1", "door", 0.9)]))
    request = AnswerRequest(document_id=document.id, message=SWISS_DOORS)
    env = {"CHATCHTG_STATE_DIR": str(tmp_path / "var"), "CHATCHTG_OBJECT_COUNTER": "detections"}

    # The Swiss kinds are off unless the setting is given: the LLM answers.
    assert build_answer_service(Settings.from_env(env), library).answer(request).model == "fake"
    everything = Settings.from_env({**env, "CHATCHTG_OBJECT_COUNTER_SKIP": ""})
    response = build_answer_service(everything, library).answer(request)
    assert response.model == OBJECT_COUNTER_MODEL
    assert response.text.startswith("Number of doors")
    skipped = Settings.from_env({**env, "CHATCHTG_OBJECT_COUNTER_SKIP": "swiss:door"})
    assert build_answer_service(skipped, library).answer(request).model == "fake"
    with pytest.raises(ValueError, match="Unknown object counter kind"):
        build_answer_service(Settings.from_env({**env, "CHATCHTG_OBJECT_COUNTER_SKIP": "swiss:bed"}), library)


def _png(path: Path, size=(40, 30)) -> bytes:
    from PIL import Image

    Image.new("L", size, 255).save(path)
    return path.read_bytes()


def _swiss_file(tmp_path: Path, thresholds: dict) -> Path:
    data = {
        "classes": [*SWISS.classes, "washing machine"],
        "score_floor": 0.05,
        "thresholds": {**thresholds, "washing machine": 0.5},
        "plans": {
            "site1_bldg1_floor1": {
                "split": "test",
                "scale": 100,
                "area": [0, 0, 40, 30],
                "detections": [
                    {"label": "door", "score": 0.9, "box": [1, 1, 5, 5]},
                    {"label": "washing machine", "score": 0.8, "box": [6, 6, 9, 9]},
                ],
            },
            "site2_bldg2_floor2": {"split": "train", "scale": 200, "area": [0, 0, 1, 1], "detections": []},
            "site3_bldg3_floor3": {"split": "valid", "scale": 100, "area": [0, 0, 1, 1], "detections": []},
        },
    }
    path = tmp_path / "detections_all.json"
    path.write_text(json.dumps(data))
    return path


def test_swiss_importer_stores_each_plan_under_the_hash_of_its_png(tmp_path):
    images = tmp_path / "swiss_dwellings"
    images.mkdir()
    first = _png(images / "site1_bldg1_floor1.png")
    _png(images / "site2_bldg2_floor2.png", (20, 10))
    store = LocalExtractionStore(tmp_path / "metadata")

    outcome = import_swiss_detections(_swiss_file(tmp_path, SWISS.thresholds), images, store)

    assert outcome == {"stored": 2, "no image": 1}
    stored = store.get(hashlib.sha256(first).hexdigest(), SWISS.producer)
    assert stored.version == SWISS.version
    assert stored.source == "data/swiss_dwellings/site1_bldg1_floor1.png"
    assert (stored.image.width, stored.image.height) == (40, 30)
    assert stored.attributes["split"] == "test" and stored.attributes["in_training_data"] is False
    assert stored.params["run_name"] == "swiss-medium-1280"
    assert [i.id for i in stored.items] == ["door-1", "washing-machine-1"]
    trained = next(e for e in store.iter(SWISS.producer, SWISS.version) if e.image.width == 20)
    assert trained.attributes["in_training_data"] is True


def test_swiss_importer_refuses_detections_with_other_minimums(tmp_path):
    images = tmp_path / "swiss_dwellings"
    images.mkdir()
    store = LocalExtractionStore(tmp_path / "metadata")
    other = {**SWISS.thresholds, "door": 0.45}

    with pytest.raises(ValueError, match="another run"):
        import_swiss_detections(_swiss_file(tmp_path, other), images, store)


def test_report_scores_each_profile_by_split_and_replays_both(tmp_path, capsys):
    from chatchtg.objects import report

    store = LocalExtractionStore(tmp_path / "metadata")
    store.put(_swiss("e" * 64, [_item("door-1", "door", 0.9), _item("door-2", "door", 0.9)], "test"))
    store.put(_extraction("d" * 64, [_item("window-1", "window", 0.99)]))
    questions = tmp_path / "questions.jsonl"
    rows = [
        {"path": "data/swiss_dwellings/site1_bldg1_floor1.png", "question": SWISS_DOORS, "answer": 2},
        {"path": "data/swiss_dwellings/site1_bldg1_floor1.png",
         "question": "Quante porte (porte interne e porte d'ingresso insieme) sono disegnate in questa pianta?",
         "answer": 3},
        {"path": "data/swiss_dwellings/site9_bldg9_floor9.png", "question": SWISS_DOORS, "answer": 1},
        {"path": "data/swiss_dwellings/site1_bldg1_floor1.png",
         "question": "How many rooms labelled 'Bad' are on this floor plan?", "answer": 1},
    ]
    questions.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    result = report.score_benchmark(questions, report.by_source(store, SWISS), SWISS)

    assert result["cells"][("door", "test")] == {"n": 2, "exact": 1, "abs_error": 1}
    assert result["missing"] == {"data/swiss_dwellings/site9_bldg9_floor9.png": 1}

    run = tmp_path / "run.jsonl"
    eval_rows = [
        {"path": "data/swiss_dwellings/site1_bldg1_floor1.png", "document_id": "e" * 64,
         "question": SWISS_DOORS, "expected": 2, "correct": False, "extracted_value": 3},
        {"path": "data/cubicasa5k/a.png", "document_id": "d" * 64,
         "question": "How many windows are shown on this floor plan?", "expected": 1, "correct": True},
        {"path": "data/cubicasa5k/a.png", "document_id": "d" * 64,
         "question": "How many saunas are there?", "expected": 1, "correct": True},
    ]
    run.write_text("\n".join(json.dumps(r) for r in eval_rows) + "\n")

    replayed = report.replay(run, store, DetectorObjectCounter(store, skip=()))

    assert (replayed["before"], replayed["after"]) == (2, 3)
    assert [(p, k, s) for _, p, k, s, *_ in replayed["replaced"]] == [
        ("swiss", "door", "test"),
        ("cubicasa", "window", "test"),
    ]
    assert "fixed" in capsys.readouterr().out


def test_no_swiss_kind_is_answered_by_default(tmp_path):
    store = LocalExtractionStore(tmp_path / "metadata")
    store.put(_swiss(items=[_item("door-1", "door", 0.9), _item("toilet-1", "toilet", 0.9)]))
    counter = DetectorObjectCounter(store)

    assert DEFAULT_SKIP == {("swiss", kind) for kind in SWISS.classes}
    assert counter.solve("e" * 64, SWISS_DOORS) is None
