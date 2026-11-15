import io
import json
from pathlib import Path

import pytest
import streamlit
from PIL import Image
from streamlit.testing.v1 import AppTest

import chatchtg.ui.app as app_module
from chatchtg.areas.pipeline import PlanAnalysis, PlanRoom
from chatchtg.config import Settings
from chatchtg import evidence as evidence_module
from chatchtg.context import areas
from conftest import EvidenceComposer
from chatchtg.schemas import AnswerResponse
from chatchtg.wiring import build_library, build_saved_answers

APP = str(Path(app_module.__file__))


@pytest.fixture(autouse=True)
def fresh_services():
    """The app caches its library for the process. Each test has its own state folder."""
    streamlit.cache_resource.clear()


def test_app_answers_a_question_about_a_registered_document(tmp_path, monkeypatch):
    monkeypatch.setenv("CHATCHTG_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("CHATCHTG_STATE_DIR", str(tmp_path / "state"))
    png = io.BytesIO()
    Image.new("RGB", (4, 4), "white").save(png, format="PNG")
    build_library(Settings.from_env()).add_upload(png.getvalue(), "plan.png")

    app = AppTest.from_file(APP).run()
    assert not app.exception

    app.segmented_control[0].set_value("Registered document").run()
    assert any("AI Assistant" in block.value for block in app.markdown)
    assert any("Drawing ready" in block.value for block in app.markdown)
    app.text_area[1].set_value("How many doors?")
    app.button[1].click().run()

    assert not app.exception
    assert any("How many doors?" in block.value for block in app.markdown)
    assert any("baseline@1" in block.value for block in app.caption)


def test_app_reports_a_document_whose_file_is_gone(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setenv("CHATCHTG_DATA_ROOT", str(data_root))
    monkeypatch.setenv("CHATCHTG_STATE_DIR", str(tmp_path / "state"))
    Image.new("RGB", (4, 4), "white").save(data_root / "plan.png")
    build_library(Settings.from_env()).register_file(data_root / "plan.png")
    (data_root / "plan.png").unlink()

    app = AppTest.from_file(APP).run()
    app.segmented_control[0].set_value("Registered document").run()

    assert not app.exception
    assert len(app.error) == 1 and "cannot be shown" in app.error[0].value


def real_png(size=(400, 300)):
    png = io.BytesIO()
    Image.new("RGB", size, "white").save(png, format="PNG")
    return png.getvalue()


def ask(app, message):
    app.segmented_control[0].set_value("Registered document").run()
    app.text_area[1].set_value(message)
    app.button[1].click().run()
    return app


def test_app_shows_the_evidence_of_an_area_answer(tmp_path, monkeypatch):
    monkeypatch.setenv("CHATCHTG_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("CHATCHTG_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("CHATCHTG_COMPOSER", "areas")
    kitchen = PlanRoom(
        number="1.01", name="Küche", area=8.127, method="measured", printed_area=None, measured_area=8.127,
        apartment="A", outdoor=False, seed=(100.0, 100.0),
        polygon=[(50.0, 50.0), (150.0, 50.0), (150.0, 150.0), (50.0, 150.0)],
    )
    plan = PlanAnalysis(scale=100, scale_read=True, pixels_per_metre=59.055, rooms=[kitchen], missing_numbers=[])
    monkeypatch.setattr(areas, "analyse_bytes", lambda content: plan)
    build_library(Settings.from_env()).add_upload(real_png(), "plan.png")

    app = ask(AppTest.from_file(APP).run(), "What is the net floor area of room 1.01 ('Küche') in m²?")

    assert not app.exception
    assert any("Evidence on plan.png" in caption for image in app.image for caption in image.captions)
    assert any("measured" in block.value for block in app.caption)            # the method in words
    rows = app.table[0].value.to_dict("records")
    assert rows[0]["Item"] == "1.01" and rows[0]["Value"] == "8.13 m²"
    assert rows[-1]["Item"] == "Total"
    assert not app.info

    app.toggle[0].set_value(False).run()                                      # the answer survives a rerun
    assert any("8.13" in block.value for block in app.markdown)
    assert all("Evidence on" not in caption for image in app.image for caption in image.captions)


def test_app_says_when_an_answer_has_no_evidence(tmp_path, monkeypatch):
    monkeypatch.setenv("CHATCHTG_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("CHATCHTG_STATE_DIR", str(tmp_path / "state"))
    build_library(Settings.from_env()).add_upload(real_png((4, 4)), "plan.png")

    app = ask(AppTest.from_file(APP).run(), "How many doors?")

    assert not app.exception
    assert any("No evidence" in block.value for block in app.info)
    assert any("LLM" in block.value for block in app.caption)
    assert not app.table


def test_app_drops_the_answer_when_another_document_is_selected(tmp_path, monkeypatch):
    monkeypatch.setenv("CHATCHTG_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("CHATCHTG_STATE_DIR", str(tmp_path / "state"))
    library = build_library(Settings.from_env())
    library.add_upload(real_png((4, 4)), "first.png")
    library.add_upload(real_png((6, 6)), "second.png")

    app = ask(AppTest.from_file(APP).run(), "How many doors?")
    assert any("How many doors?" in block.value for block in app.markdown)
    app.selectbox[0].select_index(1 - app.selectbox[0].index).run()

    assert not app.exception
    assert not any("How many doors?" in block.value for block in app.markdown)
    assert not app.info


def test_app_shows_the_plain_drawing_when_the_evidence_cannot_be_drawn(tmp_path, monkeypatch):
    monkeypatch.setenv("CHATCHTG_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("CHATCHTG_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setattr("chatchtg.context.registry.get_composer", lambda name: EvidenceComposer())
    monkeypatch.setattr("chatchtg.wiring.get_composer", lambda name: EvidenceComposer())

    def broken(content, evidence, focus=None):
        raise ValueError("a fault in the renderer")

    monkeypatch.setattr(evidence_module, "render_evidence", broken)
    build_library(Settings.from_env()).add_upload(real_png(), "plan.png")

    app = ask(AppTest.from_file(APP).run(), "Area of room 1.01?")

    assert not app.exception and not app.error
    assert any("could not be drawn" in block.value for block in app.warning)
    captions = [caption for image in app.image for caption in image.captions]
    assert len(app.image) == 1 and not any("Evidence on" in caption for caption in captions)
    assert app.table and app.expander                     # the rest of the page is still there


def test_app_shows_the_question_with_the_answer(tmp_path, monkeypatch):
    monkeypatch.setenv("CHATCHTG_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("CHATCHTG_STATE_DIR", str(tmp_path / "state"))
    build_library(Settings.from_env()).add_upload(real_png((4, 4)), "plan.png")

    app = ask(AppTest.from_file(APP).run(), "How many doors?")
    app.text_area[1].set_value("   ")                     # an empty question is not asked
    app.button[1].click().run()

    assert any('class="assistant-question">How many doors?<' in block.value for block in app.markdown)


def test_app_keeps_the_upload_when_the_document_control_is_cleared(tmp_path, monkeypatch):
    monkeypatch.setenv("CHATCHTG_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("CHATCHTG_STATE_DIR", str(tmp_path / "state"))

    app = AppTest.from_file(APP).run()
    app.segmented_control[0].set_value(None).run()         # a click on the selected option

    assert not app.exception and not app.info             # not the list of registered documents
    assert any("Open a construction drawing" in block.value for block in app.markdown)


def test_app_draws_the_marks_of_the_llm_and_warns_when_the_count_differs(tmp_path, monkeypatch):
    from chatchtg.llm.client import Location
    from chatchtg.llm.fake import FakeLLM

    monkeypatch.setenv("CHATCHTG_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("CHATCHTG_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("CHATCHTG_COMPOSER", "located")
    monkeypatch.setattr(
        "chatchtg.wiring.build_llm",
        lambda *args, **kwargs: FakeLLM(
            reply="There are 3 doors.", locations=[Location(label="door", point=(10, 10))]
        ),
    )
    build_library(Settings.from_env()).add_upload(real_png(), "plan.png")

    app = ask(AppTest.from_file(APP).run(), "How many doors?")

    assert not app.exception
    assert any("3, but 1 positions" in block.value for block in app.warning)
    assert any("counted by the LLM" in block.value for block in app.caption)
    assert any("Evidence on plan.png" in caption for image in app.image for caption in image.captions)
    assert app.table[0].value.to_dict("records")[0]["Item"] == "door 1"


def demo_set(tmp_path, monkeypatch, names=("first.png", "second.png")):
    """A demo set of drawings that exist below the data root of the test."""
    folder = tmp_path / "data" / "floorplans" / "data" / "swiss_dwellings"
    folder.mkdir(parents=True)
    entries = []
    for number, name in enumerate(names):
        (folder / name).write_bytes(real_png((8 + number, 8)))
        entries.append({
            "path": f"floorplans/data/swiss_dwellings/{name}", "title": f"Plan {number}", "note": "a note",
            "questions": [
                {"label": "Doors", "text": f"How many doors on plan {number}?", "reference": "7"},
                {"label": "Windows", "text": f"How many windows on plan {number}?", "reference": "3"},
            ],
        })
    file = tmp_path / "demo_set.json"
    file.write_text(json.dumps(entries))
    monkeypatch.setattr("chatchtg.demo.DEMO_SET", file)
    monkeypatch.setenv("CHATCHTG_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("CHATCHTG_STATE_DIR", str(tmp_path / "state"))


def labels(app):
    return [button.label for button in app.button]


def test_app_opens_on_the_demo_drawings_and_a_chip_asks_its_question(tmp_path, monkeypatch):
    demo_set(tmp_path, monkeypatch)

    app = AppTest.from_file(APP).run()

    assert not app.exception
    assert app.segmented_control[0].value == "★ Demo drawings"
    assert "1 · Plan 0" in labels(app) and "2 · Plan 1" in labels(app)
    assert any("★ Suggested" in block.value for block in app.markdown)
    assert any("DEMO" in block.value and "first.png" in block.value for block in app.markdown)
    assert "1 · Doors" in labels(app) and "2 · Windows" in labels(app)

    next(button for button in app.button if button.label == "2 · Windows").click().run()

    assert not app.exception
    assert any('class="assistant-question">How many windows on plan 0?<' in block.value for block in app.markdown)
    assert any("Benchmark answer: 3" in block.value for block in app.caption)
    kinds = {button.label: button.proto.type for button in app.button}
    assert kinds["2 · Windows"] == "primary" and kinds["1 · Doors"] == "secondary"   # the trail marks what was asked


def test_app_shows_the_saved_answer_of_a_chip_and_asks_nothing(tmp_path, monkeypatch):
    demo_set(tmp_path, monkeypatch)
    settings = Settings.from_env()
    AppTest.from_file(APP).run()                      # registers the drawings
    streamlit.cache_resource.clear()
    document = next(r for r in build_library(settings).list() if r.filename == "first.png")
    build_saved_answers(settings).save({
        (document.id, "How many windows on plan 0?"): AnswerResponse(
            text="The saved answer: 3", composer="baseline@1", model="saved-model"
        )
    })
    waits = []
    monkeypatch.setattr("chatchtg.demo.REPLAY_WAIT_SECONDS", 0.25)
    monkeypatch.setattr("time.sleep", waits.append)

    def never(self, request):
        raise AssertionError("a saved answer must not be asked again")

    app = AppTest.from_file(APP).run()
    monkeypatch.setattr("chatchtg.answering.AnswerService.answer", never)
    next(button for button in app.button if button.label == "2 · Windows").click().run()

    assert not app.exception
    assert any(block.value == "The saved answer: 3" for block in app.markdown)
    assert any("Model: saved-model" in block.value for block in app.caption)
    assert any("Benchmark answer: 3" in block.value for block in app.caption)
    assert 0.25 in waits                                # the wait before a saved answer

    # A chip without a saved answer is asked.
    monkeypatch.setattr("chatchtg.answering.AnswerService.answer", lambda self, request: AnswerResponse(
        text="A live answer", composer="baseline@1", model="live-model"
    ))
    next(button for button in app.button if button.label == "1 · Doors").click().run()
    assert not app.exception
    assert any(block.value == "A live answer" for block in app.markdown)


def test_app_changes_the_questions_with_the_demo_drawing(tmp_path, monkeypatch):
    demo_set(tmp_path, monkeypatch)
    app = AppTest.from_file(APP).run()
    next(button for button in app.button if button.label == "1 · Doors").click().run()

    next(button for button in app.button if button.label == "2 · Plan 1").click().run()

    assert not app.exception
    assert any("second.png" in block.value for block in app.markdown)
    assert not any("How many doors on plan 0?" in block.value for block in app.markdown)   # never on another drawing
    assert "1 · Doors" in labels(app)                                    # not asked on this drawing
    assert next(button for button in app.button if button.label == "1 · Doors").help == "How many doors on plan 1?"


def test_app_marks_a_demo_drawing_among_the_registered_documents(tmp_path, monkeypatch):
    demo_set(tmp_path, monkeypatch, names=("first.png",))
    build_library(Settings.from_env()).add_upload(real_png((5, 5)), "other.png")

    app = AppTest.from_file(APP).run()
    app.segmented_control[0].set_value("Registered document").run()

    assert not app.exception
    options = app.selectbox[0].options
    assert any(option.startswith("★ first.png") for option in options)
    assert any(option.startswith("other.png") for option in options)


def test_app_without_the_demo_files_has_no_demo_drawings(tmp_path, monkeypatch):
    monkeypatch.setenv("CHATCHTG_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("CHATCHTG_STATE_DIR", str(tmp_path / "state"))

    app = AppTest.from_file(APP).run()

    assert not app.exception
    assert app.segmented_control[0].options == ["Upload an image", "Registered document"]
    assert not any("Suggested" in block.value for block in app.markdown)
