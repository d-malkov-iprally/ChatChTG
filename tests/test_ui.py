import io
from pathlib import Path

import pytest
import streamlit
from PIL import Image
from streamlit.testing.v1 import AppTest

import chatchtg.ui.app as app_module
from chatchtg.config import Settings
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
