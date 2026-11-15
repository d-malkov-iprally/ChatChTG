"""The area composer, with the pipeline replaced by a fixed analysis."""

import pytest

from chatchtg.areas.pipeline import PlanAnalysis, PlanRoom
from chatchtg.areas.questions import parse_question
from chatchtg.context import areas
from chatchtg.context.areas import AREA_SYSTEM_PROMPT, GAP_DOUBT, SCALE_DOUBT, AreaComposer, decide
from chatchtg.context.baseline import SYSTEM_PROMPT
from chatchtg.context.composer import ComposerInput
from chatchtg.context.registry import get_composer
from chatchtg.context.types import ImagePart, TextPart
from chatchtg.evals.scoring import extract_number
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


ROOM = "What is the net floor area of room 1.01 ('Küche') in m²?"
APARTMENT = "What is the net floor area of apartment A in m²?"


def test_decide_names_the_basis_and_the_doubts():
    result = analysis(
        [room("1.01", "Küche", 8.0), room("1.02", "Zimmer", 12.0, method="printed")], missing=["1.03"]
    )
    result.scale = 200

    decision = decide(parse_question(ROOM), result)

    assert decision.confident and decision.basis == "measured"
    assert set(decision.doubts) == {SCALE_DOUBT, GAP_DOUBT}


def test_decide_finds_no_doubt_on_a_clean_plan_and_printed_basis_for_printed_rooms():
    result = analysis([room("1.02", "Zimmer", 12.0, method="printed")])

    decision = decide(parse_question("What is the net floor area of room 1.02 ('Zimmer') in m²?"), result)

    assert decision.basis == "printed" and decision.doubts == ()


def test_decide_has_no_basis_and_no_confidence_without_a_value():
    decision = decide(parse_question(ROOM), analysis([room("1.02", "Zimmer", 12.0)]))

    assert decision.answer.value is None and not decision.confident and decision.basis is None


def test_a_text_only_answer_carries_the_computed_value_and_the_statement(library, plan):
    plan(analysis([room("1.01", "Küche", 8.127), room("1.02", "Zimmer 2", 12.5, method="printed")]))

    computed = compose(library, APARTMENT).computed

    assert computed.value == pytest.approx(20.627) and computed.basis == "measured" and computed.doubts == []
    text = computed.statement
    assert text.startswith("Sum of the 2 indoor rooms of apartment A: 1.01, 1.02.\n\n")
    assert "- 1.01 Küche: 8.13 m² (measured from the drawn lines)" in text
    assert "- 1.02 Zimmer 2: 12.50 m² (printed on the plan)" in text
    assert "without a language model" in text and "Scale of the drawing: 1:100." in text
    assert "A measured area can differ from the true area by a few percent." in text
    assert text.endswith("\n\nAnswer: 20.63 m²") and extract_number(text) == 20.63


def test_the_statement_of_a_single_room_ends_with_its_area_not_its_number(library, plan):
    plan(analysis([room("1.01", "Küche", 8.127, method="printed")]))

    text = compose(library, ROOM).computed.statement

    assert extract_number(text) == 8.13 and "few percent" not in text and "Caution" not in text


def test_the_statement_names_the_doubts(library, plan):
    result = analysis([room("1.01", "Küche", 8.0), room("1.02", "Zimmer", 12.0)], missing=["1.03"])
    result.scale = 200
    plan(result)

    context = compose(library, APARTMENT)

    assert set(context.computed.doubts) == {SCALE_DOUBT, GAP_DOUBT}
    assert "measured areas are less reliable" in context.computed.statement
    assert "gap (1.03)" in context.computed.statement
    assert extract_number(context.computed.statement) == 20.0


def test_a_coarse_scale_is_no_caution_for_printed_areas(library, plan):
    result = analysis([room("1.01", "Küche", 8.0, method="printed")])
    result.scale = 200
    plan(result)

    computed = compose(library, ROOM).computed

    assert computed.doubts == [SCALE_DOUBT] and "Caution" not in computed.statement


def test_a_gap_is_no_caution_for_a_single_room(library, plan):
    plan(analysis([room("1.01", "Küche", 8.0)], missing=["1.03"]))

    assert "gap" not in compose(library, ROOM).computed.statement


def test_the_statement_lists_a_shared_region_once(library, plan):
    hall, corridor = room("1.01", "Zimmer", 15.0), room("1.02", "Korridor", None)
    hall.region = corridor.region = "1.01"
    plan(analysis([hall, corridor]))

    text = compose(library, APARTMENT).computed.statement

    assert "- 1.02 Korridor: included in the area of 1.01" in text and extract_number(text) == 15.0


def test_the_statement_does_not_name_a_scale_that_was_guessed(library, plan):
    result = analysis([room("1.01", None, 8.127, method="printed")])
    result.scale_read = False
    plan(result)

    text = compose(library, ROOM).computed.statement

    assert "Scale of the drawing" not in text and "- 1.01 no name: 8.13 m² (printed on the plan)" in text


def test_a_question_that_goes_to_the_llm_has_no_computed_answer(library, plan):
    plan(analysis([room("1.02", "Zimmer", 12.5)]))

    context = compose(library, ROOM)  # room 1.01 was not found: image plus room table

    assert context.computed is None and "computed" not in context.summary()


def test_the_summary_has_the_computed_value_without_the_statement(library, plan):
    plan(analysis([room("1.01", "Küche", 8.127)]))

    summary = compose(library, ROOM).summary()["computed"]

    assert summary == {"value": pytest.approx(8.127), "unit": "m2", "basis": "measured", "doubts": []}


def test_a_room_number_read_twice_is_described_by_the_reading_that_has_the_area(library, plan):
    plan(analysis([room("1.01", "Küche", 8.0, method="printed"), room("1.01", "Kuche", None)]))

    computed = compose(library, ROOM).computed

    assert computed.basis == "printed"
    assert "- 1.01 Küche: 8.00 m² (printed on the plan)" in computed.statement
    assert "few percent" not in computed.statement


@pytest.mark.parametrize("order", [("printed", "measured"), ("measured", "printed")])
def test_a_total_describes_each_reading_of_a_number_read_twice_by_its_own_method(library, plan, order):
    areas_by_method = {"printed": 8.0, "measured": 9.0}
    rooms = [room("1.01", "Zimmer", areas_by_method[method], method=method) for method in order]
    result = analysis(rooms)
    result.scale_read = False
    plan(result)
    query = parse_question(APARTMENT)

    decision = decide(query, result)
    context = compose(library, APARTMENT)

    # A measured area on a guessed scale: the LLM gets the image.
    assert decision.basis == "measured" and not decision.confident
    assert context.computed is None
    text = areas.statement_text(query, result, decision)
    assert "- 1.01 Zimmer: 8.00 m² (printed on the plan)" in text
    assert "- 1.01 Zimmer: 9.00 m² (measured from the drawn lines)" in text
    assert "few percent" in text
    table = areas.result_text(result, decision)
    assert "8.00 m² (printed)" in table and "9.00 m² (measured)" in table


@pytest.mark.parametrize(
    ("question", "total"),
    [
        ("What is the total net floor area of all rooms labelled 'Zimmer 2' on this floor in m²?", 12.5),
        ("What is the total net floor area of all numbered indoor rooms on this floor in m²?", 20.63),
    ],
)
def test_the_statement_of_a_label_or_floor_total_ends_with_the_total(library, plan, question, total):
    rooms = [room("1.01", "Küche", 8.127), room("1.02", "Zimmer 2", 12.5), room("1.03", "Balkon 3", 4.0, outdoor=True)]
    plan(analysis(rooms, missing=["1.04"]))

    text = compose(library, question).computed.statement

    assert extract_number(text) == total and text.endswith(f"Answer: {total:.2f} m²") and "gap (1.04)" in text
