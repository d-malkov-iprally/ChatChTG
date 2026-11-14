import cv2
import numpy as np
import pytest

from chatchtg.answering import ROOM_COUNTER_MODEL, AnswerService
from chatchtg.context.baseline import BaselineComposer
from chatchtg.evals.scoring import score
from chatchtg.llm.fake import FakeLLM
from chatchtg.rooms import geometry, lexicon
from chatchtg.rooms.questions import RoomQuestion, classify
from chatchtg.rooms.router import Finding, decide
from chatchtg.schemas import AnswerRequest
from conftest import PNG


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("How many bedrooms (makuuhuone, usually marked MH) are on this floor plan?", RoomQuestion("bedroom")),
        ("Kuinka monta erillistä WC-tilaa pohjapiirustuksessa on?", RoomQuestion("wc")),
        ("How many floors (storeys) does this floor plan show?", RoomQuestion("storeys")),
        ("Wie viele Wohnungen sind auf diesem Grundriss dargestellt?", RoomQuestion("apartments")),
        ("Combien d'appartements (logements) figurent sur ce plan d'étage ?", RoomQuestion("apartments")),
        (
            "How many rooms labelled 'Cage d'escalier' are there on this floor?",
            RoomQuestion("label", "Cage d'escalier"),
        ),
        (
            "Quanti locali con la denominazione «Vano scala» ci sono in questo piano?",
            RoomQuestion("label", "Vano scala"),
        ),
        ("How many windows are shown on this floor plan?", None),
        ("How many toilet bowls (WC seats) are drawn on this floor plan?", None),
        ("Wie gross ist die Nettofläche von Raum 0.27 («Treppenhaus») in m²?", None),
    ],
)
def test_classify_recognises_room_count_questions_only(question, expected):
    assert classify(question) == expected


def token(text, x, y, conf=1.0, **more):
    return {"text": text, "conf": conf, "box": [x, y, x + 40, y + 12], **more}


def plan(*tokens, height=1000):
    return {"W": 1000, "H": height, "tokens": list(tokens)}


def test_finnish_labels_are_counted_once_per_place():
    read = plan(
        token("MH", 100, 100),
        token("MH", 102, 101, conf=0.6),  # the same label, read again in another pass
        token("MH 12,5 m2", 400, 100),
        token("OH", 400, 400),
        token("KH", 700, 100),
        token("WC", 700, 400),
        token("KPH/WC", 100, 700),
    )

    assert lexicon.count_cubi(read, "bedroom")[0] == 2
    assert lexicon.count_cubi(read, "bathroom")[0] == 2
    assert lexicon.count_cubi(read, "wc")[0] == 1  # the WC of "KPH/WC" is not a separate room


def test_single_letters_count_only_among_room_words():
    assert lexicon.count_cubi(plan(token("S", 100, 100)), "sauna")[0] == 1
    assert lexicon.count_cubi(plan(token("S 4,5 m2", 100, 100)), "sauna")[0] == 1
    assert lexicon.count_cubi(plan(token("S Virtanen", 100, 100)), "sauna")[0] == 0


def test_apartment_type_string_reveals_a_sauna():
    read = plan(token("3H+K+S 78,5 m2", 100, 900))

    assert lexicon.count_cubi(read, "sauna")[0] == 1
    assert lexicon.count_cubi(read, "kitchen")[0] == 1


def test_swiss_label_must_match_whole():
    read = plan(
        token("Zimmer", 100, 100, src="a"),
        token("Schlafzimmer", 300, 100, src="a"),
        token("Zimner", 500, 100, src="a"),  # a misread "Zimmer"
        token("Wohnen/Essen", 100, 300, src="a"),
        token("Zimmer", 100, 950, src="a"),  # in the legend strip at the bottom
    )

    assert len(lexicon.swiss_label_hits(read, "Zimmer")) == 2
    assert len(lexicon.swiss_label_hits(read, "Wohnen")) == 0
    assert len(lexicon.swiss_label_hits(read, "Wohnen/Essen")) == 1


def test_swiss_apartments_follow_the_unit_letters():
    read = plan(token("Whg. A", 100, 100), token("Whg. C", 500, 100))

    assert lexicon.swiss_apartments_count(read)[0] == 3


def test_storeys_are_separate_hollow_drawings():
    sheet = np.full((600, 1000, 3), 255, np.uint8)
    cv2.rectangle(sheet, (50, 100), (400, 500), (0, 0, 0), 6)
    cv2.rectangle(sheet, (550, 100), (950, 500), (0, 0, 0), 6)
    cv2.line(sheet, (550, 300), (950, 300), (0, 0, 0), 4)

    assert geometry.count_storeys(sheet)[0] == 2


def test_apartments_are_the_areas_inside_red_outlines():
    sheet = np.full((600, 1000, 3), 255, np.uint8)
    for left in (60, 520):
        cv2.rectangle(sheet, (left - 14, 86), (left + 414, 514), (175, 175, 175), 12)  # the walls
        cv2.rectangle(sheet, (left, 100), (left + 400, 500), (0, 0, 255), 3)  # the unit outline

    count, boxes = geometry.count_apartments(sheet)

    assert count == 2
    assert len(boxes) == 2


def test_decide_falls_back_to_the_usual_answer_when_nothing_was_read():
    unread = {"labels": 0, "room_words": 0, "typology": None}
    read = {"labels": 3, "room_words": 6, "typology": None}

    assert decide("bedroom", unread) == (1, "usual", False)
    assert decide("bedroom", read) == (3, "labels", False)
    assert decide("sauna", {"labels": 0, "room_words": 6, "typology": ["K", "S"]})[0] == 1


def test_only_apartments_confirmed_by_two_methods_are_confident():
    assert decide("apartments", {"labels": 4, "shapes": 4}) == (4, "labels+shapes", True)
    assert decide("apartments", {"labels": 4, "shapes": 3}) == (4, "labels", False)
    assert decide("apartments", {"labels": 0, "shapes": 3}) == (3, "shapes", False)
    assert decide("label", {"labels": 2})[2] is False


class FixedCounter:
    def __init__(self, finding):
        self.finding = finding

    def solve(self, content, message):
        return self.finding


def finding(confident):
    evidence = [{"text": "MH", "box": [1, 2, 3, 4]}, {"text": "MH 2", "box": [5, 6, 7, 8]}]
    return Finding(RoomQuestion("bedroom"), 2, "labels", confident, evidence)


def test_confident_room_count_is_answered_without_the_llm(library):
    document = library.add_upload(PNG, "plan.png")
    service = AnswerService(library, BaselineComposer(), FakeLLM(reply="7"), FixedCounter(finding(True)))

    trace = service.answer_traced(AnswerRequest(document_id=document.id, message="How many bedrooms?"))

    assert trace.response.model == ROOM_COUNTER_MODEL
    assert trace.response.method == "labels"
    assert trace.response.evidence[0] == {"text": "MH", "box": [1, 2, 3, 4]}
    assert score("count", 2, trace.response.text) is True


def test_unsure_room_count_goes_to_the_llm_unless_trusted(library):
    document = library.add_upload(PNG, "plan.png")
    request = AnswerRequest(document_id=document.id, message="How many bedrooms?")
    unsure = FixedCounter(finding(False))

    asked = AnswerService(library, BaselineComposer(), FakeLLM(reply="7"), unsure).answer(request)
    trusted = AnswerService(
        library, BaselineComposer(), FakeLLM(reply="7"), unsure, trust_room_counter=True
    ).answer(request)

    assert (asked.text, asked.model, asked.method) == ("7", "fake", None)
    assert trusted.model == ROOM_COUNTER_MODEL


def test_other_questions_go_to_the_llm(library):
    document = library.add_upload(PNG, "plan.png")
    service = AnswerService(library, BaselineComposer(), FakeLLM(reply="7"), FixedCounter(None))

    response = service.answer(AnswerRequest(document_id=document.id, message="How many doors?"))

    assert response.text == "7"


def test_room_conventions_are_added_to_finnish_room_counts_only(library):
    from chatchtg.context.baseline import BaselineComposer as Baseline
    from chatchtg.context.composer import ComposerInput
    from chatchtg.context.room_conventions import RoomConventionsComposer

    document = library.add_upload(PNG, "plan.png")

    def compose(composer, message):
        return composer.compose(ComposerInput(document=document, content=PNG, message=message))

    kitchens = "How many kitchens (including kitchenettes) are on this floor plan?"
    context = compose(RoomConventionsComposer(), kitchens)
    assert "Do not count" in context.parts[1].text  # the neighbouring home
    assert context.parts[-1].text == kitchens
    storeys = compose(RoomConventionsComposer(), "How many floors (storeys) does this floor plan show?")
    assert "outbuilding" in storeys.parts[1].text
    for other in (
        "How many windows are shown on this floor plan?",
        "How many apartments (residential units) are shown on this floor plan?",
        "How many rooms labelled 'Bad' are there on this floor?",
    ):
        assert compose(RoomConventionsComposer(), other) == compose(Baseline(), other)


def test_a_misread_unit_tag_is_not_one_more_apartment():
    assert lexicon.swiss_apartments_count(plan(token("Whg. A", 100, 100), token("Whg. Al", 500, 100)))[0] == 1
    read = plan(token("App. A", 100, 100), token("App. B", 500, 100), token("App. t", 100, 500))
    assert lexicon.swiss_apartments_count(read)[0] == 2


def test_a_swiss_label_read_in_two_tiles_is_counted_once():
    read = plan(token("Bad", 100, 100, src="a"), token("Bad", 101, 101, src="a"))

    assert len(lexicon.swiss_label_hits(read, "Bad")) == 1


def test_glazed_alone_is_not_a_balcony():
    assert lexicon.count_cubi(plan(token("LASITETTU TERASSI", 100, 100), token("OH", 400, 400)), "balcony")[0] == 0
    assert lexicon.count_cubi(plan(token("LASITETTU PARVEKE", 100, 100), token("OH", 400, 400)), "balcony")[0] == 1


def test_a_dressing_room_is_not_a_bathroom():
    assert not lexicon.fuzzy_word("PUKUHUONE", "bathroom")
    assert lexicon.fuzzy_word("PESUHUONF", "bathroom")


def test_the_type_string_is_read_with_a_cyrillic_three():
    assert lexicon.typology(plan(token("ЗН+K+S", 100, 900)))["rooms"] == 3


def test_a_cut_off_cache_file_is_read_again(tmp_path, monkeypatch):
    import hashlib
    import io

    from PIL import Image

    from chatchtg.rooms import ocr

    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), "white").save(buffer, format="PNG")
    content = buffer.getvalue()
    monkeypatch.setattr(ocr, "_read_swiss", lambda image: {"tokens": []})
    cache_file = tmp_path / f"{hashlib.sha256(content).hexdigest()}-swiss-v{ocr._VERSION}.json"
    cache_file.write_text('{"tokens": [')

    assert ocr.read_plan(content, "swiss", tmp_path)["tokens"] == []
    assert ocr.read_plan(content, "swiss", tmp_path)["W"] == 8
    assert [path.name for path in tmp_path.iterdir()] == [cache_file.name]
