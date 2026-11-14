import io
from pathlib import Path

import pytest
import streamlit
from PIL import Image
from streamlit.testing.v1 import AppTest

import chatchtg.ui.app as app_module
from chatchtg.areas.pipeline import PlanAnalysis, PlanRoom
from chatchtg.config import Settings
from chatchtg.context import areas
from chatchtg.wiring import build_library

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

    app.radio[0].set_value("Registered document").run()
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
    app.radio[0].set_value("Registered document").run()

    assert not app.exception
    assert len(app.error) == 1 and "cannot be shown" in app.error[0].value


def real_png(size=(400, 300)):
    png = io.BytesIO()
    Image.new("RGB", size, "white").save(png, format="PNG")
    return png.getvalue()


def ask(app, message):
    app.radio[0].set_value("Registered document").run()
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
