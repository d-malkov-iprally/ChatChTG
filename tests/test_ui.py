import io
from pathlib import Path

from PIL import Image

from streamlit.testing.v1 import AppTest

import chatchtg.ui.app as app_module
from chatchtg.config import Settings
from chatchtg.wiring import build_library

APP = str(Path(app_module.__file__))


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
