import io
from pathlib import Path

from PIL import Image
from streamlit.testing.v1 import AppTest

import chatchtg.ui
from chatchtg.cli import main
from chatchtg.detection import MODELS, Detection, count_by_label, draw_detections, parse_labels
from chatchtg.detection import models

# Not imported as a module: an import runs the script outside Streamlit, and its open form
# then breaks the forms of the other app's test.
APP = str(Path(chatchtg.ui.__file__).parent / "detect_app.py")


def test_parse_labels_splits_and_normalises_the_prompt():
    assert parse_labels(" Window, door.  Sliding   Door\nwindow,, ") == [
        "window",
        "door",
        "sliding door",
    ]
    assert parse_labels(" , . ") == []


def test_count_by_label_puts_the_most_frequent_label_first():
    detections = [
        Detection("door", 0.9, (0, 0, 1, 1)),
        Detection("window", 0.8, (0, 0, 1, 1)),
        Detection("window", 0.7, (0, 0, 1, 1)),
    ]
    assert count_by_label(detections) == {"window": 2, "door": 1}


def test_draw_detections_draws_the_box_and_keeps_the_original():
    image = Image.new("RGB", (200, 100), "white")
    annotated = draw_detections(image, [Detection("door", 0.5, (50, 40, 150, 90))])

    assert annotated.size == image.size
    assert annotated.getpixel((50, 60)) != (255, 255, 255)  # the left side of the box
    assert annotated.getpixel((100, 65)) == (255, 255, 255)  # inside the box
    assert image.getpixel((50, 60)) == (255, 255, 255)


def test_every_model_has_a_known_family():
    assert {spec.family for spec in MODELS.values()} == {"grounding-dino", "owlv2"}


def test_cli_has_the_detect_demo_command(monkeypatch):
    calls = []
    monkeypatch.setattr("chatchtg.cli.subprocess.call", lambda command: calls.append(command) or 0)

    assert main(["detect-demo"]) == 0
    assert calls[0][-1].endswith("detect_app.py")
    assert "--server.address=127.0.0.1" in calls[0]


class FakeDetector:
    def __init__(self):
        self.calls = []

    def detect(self, image, labels, threshold):
        self.calls.append((image.size, labels, threshold))
        return [Detection(labels[0], 0.8, (1, 1, 20, 20)), Detection(labels[0], 0.6, (30, 30, 50, 50))]


def test_app_detects_the_named_objects_on_a_dataset_image(tmp_path, monkeypatch):
    Image.new("RGB", (64, 48), "white").save(tmp_path / "plan.png")
    monkeypatch.setenv("CHATCHTG_DATA_ROOT", str(tmp_path))
    fake = FakeDetector()
    monkeypatch.setattr(models, "load_detector", lambda name: fake)

    app = AppTest.from_file(APP, default_timeout=30).run()
    assert not app.exception

    app.segmented_control(key="source").set_value("Dataset folder").run()
    app.segmented_control(key="model").set_value("OWLv2 base").run()
    app.text_input(key="prompt").set_value("Door, toilet")
    app.button[0].click().run()

    assert not app.exception
    assert fake.calls == [((64, 48), ["door", "toilet"], 0.1)]
    assert app.metric[0].value == "2"
    assert any("OWLv2 base" in caption.value for caption in app.caption)


def test_app_refuses_a_folder_outside_the_data_folder(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    data_root.mkdir()
    Image.new("RGB", (8, 8), "white").save(tmp_path / "secret.png")
    (data_root / "link.png").symlink_to(tmp_path / "secret.png")
    monkeypatch.setenv("CHATCHTG_DATA_ROOT", str(data_root))

    app = AppTest.from_file(APP, default_timeout=30).run()
    app.segmented_control(key="source").set_value("Dataset folder").run()

    # A file of the data folder that is a link to a file outside it.
    assert not app.exception
    assert [error.value for error in app.error] == [
        "This file is a link to a file outside the data folder."
    ]
    assert app.button[0].disabled

    for folder in ["..", str(tmp_path)]:
        app.text_input(key="folder").set_value(folder).run()
        assert not app.exception
        assert [error.value for error in app.error] == ["The folder must be inside the data folder."]
        assert not app.selectbox
        assert app.button[0].disabled
