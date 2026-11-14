"""The area composer, with the pipeline replaced by a fixed analysis."""

import pytest

from chatchtg.areas.pipeline import PlanAnalysis, PlanRoom
from chatchtg.context import areas
from chatchtg.context.areas import AREA_SYSTEM_PROMPT, AreaComposer
from chatchtg.context.baseline import SYSTEM_PROMPT
from chatchtg.context.composer import ComposerInput
from chatchtg.context.registry import get_composer
from chatchtg.context.types import ImagePart, TextPart
from conftest import PNG


def room(number, name, area, apartment="A", outdoor=False, method="measured"):
    return PlanRoom(
        number=number, name=name, area=area, method=method if area is not None else None,
        printed_area=None, measured_area=area, apartment=apartment, outdoor=outdoor, seed=(0.0, 0.0),
    )


def analysis(rooms, missing=()):
    return PlanAnalysis(
        scale=100, scale_read=True, pixels_per_metre=59.055, rooms=rooms, missing_numbers=list(missing)
    )


@pytest.fixture
def plan(monkeypatch):
    """Set the analysis that the composer will get for any image."""

    def use(result):
        if isinstance(result, Exception):
            def fail(content):
                raise result
            monkeypatch.setattr(areas, "analyse_bytes", fail)
        else:
            monkeypatch.setattr(areas, "analyse_bytes", lambda content: result)

    return use


def compose(library, message):
    document = library.add_upload(PNG, "plan.png")
    return AreaComposer().compose(ComposerInput(document=document, content=PNG, message=message))


def test_the_area_composer_is_registered():
    assert get_composer("areas").name == "areas"


def test_an_area_question_gets_the_computed_result_and_no_image(library, plan):
    plan(analysis([room("1.01", "Küche", 8.127), room("1.02", "Zimmer", 12.5)]))

    context = compose(library, "What is the net floor area of room 1.01 ('Küche') in m²?")

    assert context.system == AREA_SYSTEM_PROMPT
    assert not any(isinstance(part, ImagePart) for part in context.parts)
    assert "Result: 8.13 m²" in context.parts[0].text
    assert "1.01 Küche: 8.13 m² (measured)" in context.parts[0].text
    assert context.parts[-1] == TextPart(text="What is the net floor area of room 1.01 ('Küche') in m²?")


def test_a_question_that_is_not_about_area_is_composed_as_in_the_baseline(library, plan):
    plan(RuntimeError("the pipeline must not run"))

    context = compose(library, "How many doors?")

    assert context.system == SYSTEM_PROMPT
    assert context.parts == [ImagePart(media_type="image/png", data=PNG), TextPart(text="How many doors?")]


def test_a_room_that_was_not_found_falls_back_to_the_image_with_the_room_table(library, plan):
    plan(analysis([room("1.02", "Zimmer", 12.5)]))

    context = compose(library, "What is the net floor area of room 1.01 ('Küche') in m²?")

    assert context.system == SYSTEM_PROMPT
    assert isinstance(context.parts[0], ImagePart)
    assert "1.02 Zimmer: 12.50 m² (measured)" in context.parts[1].text


def test_a_total_with_a_gap_in_the_numbering_falls_back_on_a_plan_that_prints_areas(library, plan):
    rooms = [room("1.01", "Zimmer", 10.0, method="printed"), room("1.03", "Zimmer", 12.0, method="printed")]
    for r in rooms:
        r.printed_area = r.area
    plan(analysis(rooms, missing=["1.02"]))

    context = compose(library, "What is the total net floor area of all rooms labelled 'Zimmer' on this floor in m²?")

    assert isinstance(context.parts[0], ImagePart)
    assert "did not find: 1.02" in context.parts[1].text


def test_a_total_with_a_gap_in_the_numbering_is_still_answered_on_a_plan_without_printed_areas(library, plan):
    plan(analysis([room("1.01", "Zimmer", 10.0), room("1.03", "Zimmer", 12.0)], missing=["1.02"]))

    context = compose(library, "What is the total net floor area of all rooms labelled 'Zimmer' on this floor in m²?")

    assert context.system == AREA_SYSTEM_PROMPT
    assert "Result: 22.00 m²" in context.parts[0].text


def test_a_single_room_is_answered_even_when_another_room_is_missing(library, plan):
    plan(analysis([room("1.01", "Zimmer", 10.0)], missing=["1.02"]))

    context = compose(library, "What is the net floor area of room 1.01 ('Zimmer') in m²?")

    assert context.system == AREA_SYSTEM_PROMPT
    assert "Result: 10.00 m²" in context.parts[0].text


def test_an_image_the_pipeline_cannot_read_is_composed_as_in_the_baseline(library, plan):
    plan(ValueError("not an image"))

    context = compose(library, "What is the net floor area of room 1.01 ('Küche') in m²?")

    assert context.system == SYSTEM_PROMPT
    assert isinstance(context.parts[0], ImagePart)


def test_a_measured_area_with_a_guessed_scale_goes_to_the_llm_with_the_image(library, plan):
    result = analysis([room("1.01", "Küche", 8.127)])
    result.scale_read = False
    plan(result)

    context = compose(library, "What is the net floor area of room 1.01 ('Küche') in m²?")

    assert context.system == SYSTEM_PROMPT
    assert isinstance(context.parts[0], ImagePart)
    assert "scale not read, 1:100 assumed" in context.parts[1].text


def test_a_printed_area_does_not_need_the_scale(library, plan):
    result = analysis([room("1.01", "Küche", 8.127, method="printed")])
    result.scale_read = False
    plan(result)

    context = compose(library, "What is the net floor area of room 1.01 ('Küche') in m²?")

    assert context.system == AREA_SYSTEM_PROMPT


def test_rooms_measured_as_one_region_are_listed_once(library, plan):
    hall, corridor = room("1.01", "Zimmer", 15.0), room("1.02", "Korridor", None)
    hall.region = corridor.region = "1.01"
    plan(analysis([hall, corridor, room("1.03", "Bad", 5.0)]))

    context = compose(library, "What is the net floor area of apartment A in m²?")

    assert context.system == AREA_SYSTEM_PROMPT
    assert "1.02 Korridor: included in the area of 1.01" in context.parts[0].text
    assert "Result: 20.00 m²" in context.parts[0].text


def test_a_failure_of_the_pipeline_is_logged(library, plan, caplog):
    plan(RuntimeError("broken"))

    context = compose(library, "What is the net floor area of room 1.01 ('Küche') in m²?")

    assert context.system == SYSTEM_PROMPT
    assert "The area pipeline failed on plan.png" in caplog.text


def test_only_the_first_room_of_a_shared_region_keeps_the_measured_area():
    from chatchtg.areas.pipeline import _join_shared_regions

    rooms = [room("1.01", "Zimmer", 15.0), room("1.02", "Korridor", 10.0), room("1.03", "Bad", 5.0)]
    _join_shared_regions(rooms, [None, 0, None])

    assert [(r.area, r.method, r.region) for r in rooms] == [
        (15.0, "measured", "1.01"), (None, None, "1.01"), (5.0, "measured", None),
    ]

    rooms = [room("1.01", "Zimmer", 15.0, method="printed"), room("1.02", "Korridor", 10.0)]
    _join_shared_regions(rooms, [None, 0])

    # The first room prints its own area, so the region's area cannot stand for both.
    assert [(r.area, r.method, r.region) for r in rooms] == [(15.0, "printed", None), (None, None, None)]


QUESTION_A = "What is the net floor area of apartment A in m²?"


def test_an_answered_area_question_carries_the_rooms_as_evidence(library, plan):
    kitchen, bedroom = room("1.01", "Küche", 8.127), room("1.02", "Zimmer", 12.5, method="printed")
    kitchen.polygon = [(10.4, 10.6), (90.0, 10.0), (90.0, 60.0), (10.0, 60.0)]
    bedroom.polygon = [(100.0, 10.0), (180.0, 10.0), (180.0, 60.0)]
    plan(analysis([kitchen, bedroom, room("1.03", "Balkon", 4.0, outdoor=True)]))

    context = compose(library, QUESTION_A)

    assert context.method == "printed+measured"
    assert context.evidence == [
        {"text": "1.01", "detail": "Küche", "kind": "measured", "value": 8.13, "unit": "m²",
         "polygon": [[10, 11], [90, 10], [90, 60], [10, 60]]},
        {"text": "1.02", "detail": "Zimmer", "kind": "printed", "value": 12.5, "unit": "m²",
         "polygon": [[100, 10], [180, 10], [180, 60]]},
        {"text": "Total", "detail": "Sum of the 2 indoor rooms of apartment A: 1.01, 1.02.",
         "kind": "total", "value": 20.63, "unit": "m²"},
    ]


def test_rooms_measured_as_one_region_are_one_evidence_item(library, plan):
    hall, corridor = room("1.01", "Zimmer", 15.0), room("1.02", "Korridor", None)
    hall.region = corridor.region = "1.01"
    hall.polygon = [(0.0, 0.0), (50.0, 0.0), (50.0, 30.0), (0.0, 30.0)]
    corridor.polygon = [(20.0, 0.0), (50.0, 0.0), (50.0, 30.0)]
    plan(analysis([hall, corridor, room("1.03", "Bad", 5.0)]))

    shared, bath, total = compose(library, QUESTION_A).evidence

    assert shared == {
        "text": "1.01+1.02", "detail": "Zimmer, Korridor: measured as one region, no wall was found between them",
        "kind": "shared", "value": 15.0, "unit": "m²", "polygon": [[0, 0], [50, 0], [50, 30], [0, 30]],
    }
    assert bath["text"] == "1.03" and total["value"] == 20.0


def test_a_room_without_an_outline_is_marked_at_its_label(library, plan):
    kitchen = room("1.01", "Küche", 8.0, method="printed")
    kitchen.seed = (40.6, 20.2)
    plan(analysis([kitchen]))

    item = compose(library, "What is the net floor area of room 1.01 ('Küche') in m²?").evidence[0]

    assert item["point"] == [41, 20] and "polygon" not in item


def test_a_question_the_pipeline_cannot_answer_carries_no_evidence(library, plan):
    plan(analysis([room("1.02", "Zimmer", 12.5)]))

    context = compose(library, "What is the net floor area of room 1.01 ('Küche') in m²?")

    assert context.method is None and context.evidence == []
