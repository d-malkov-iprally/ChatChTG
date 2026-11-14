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
