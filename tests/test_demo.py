import io
import json

from PIL import Image

from chatchtg.cli import main
from chatchtg.config import Settings
from chatchtg.demo import DemoDrawing, DemoQuestion, SavedAnswers, load_demo_set, register_demo_set
from chatchtg.schemas import AnswerResponse
from chatchtg.wiring import build_saved_answers


def png(size=(4, 4)):
    data = io.BytesIO()
    Image.new("RGB", size, "white").save(data, format="PNG")
    return data.getvalue()


def drawing(path):
    return DemoDrawing(path=path, title="t", note="n", questions=(DemoQuestion(label="l", text="q"),))


def test_the_demo_set_of_the_package_is_complete():
    drawings = load_demo_set()

    assert drawings
    assert len({d.path for d in drawings}) == len(drawings)
    for d in drawings:
        assert d.path.startswith("floorplans/data/") and d.title and d.questions
        assert all(q.label and q.text and q.reference for q in d.questions)
        assert len({q.label for q in d.questions}) == len(d.questions)   # a chip names one question


def test_load_demo_set_reads_a_file(tmp_path):
    file = tmp_path / "set.json"
    file.write_text(json.dumps([
        {"path": "a/data/s/x.png", "title": "X", "questions": [{"label": "Doors", "text": "How many doors?"}]}
    ]))

    assert load_demo_set(file) == [
        DemoDrawing(path="a/data/s/x.png", title="X", note="", questions=(DemoQuestion("Doors", "How many doors?"),))
    ]


def test_register_demo_set_registers_the_files_that_exist(library, data_root):
    folder = data_root / "floorplans" / "data" / "swiss_dwellings"
    folder.mkdir(parents=True)
    (folder / "here.png").write_bytes(png())
    here = drawing("floorplans/data/swiss_dwellings/here.png")
    gone = drawing("floorplans/data/swiss_dwellings/gone.png")

    registered = register_demo_set(library, data_root, [gone, here])

    assert [d for d, _ in registered] == [here]
    record = registered[0][1]
    assert library.get(record.id).filename == "here.png"
    assert record.attributes == {                       # as `chatchtg ingest` gives them
        "original_path": "data/swiss_dwellings/here.png",
        "source": "swiss_dwellings",
        "licence_file": "data/swiss_dwellings/LICENSE.md",
    }
    assert library.count() == 1


def test_register_demo_set_leaves_out_a_path_outside_the_data_root(library, data_root, tmp_path):
    (tmp_path / "outside.png").write_bytes(png())
    (data_root / "notes.txt").write_text("not an image")

    assert register_demo_set(library, data_root, [drawing("../outside.png"), drawing("notes.txt")]) == []
    assert library.count() == 0


def response(text="Answer: 7"):
    return AnswerResponse(
        text=text, composer="baseline@1", model="fake", method="printed",
        evidence=[{"text": "1.01", "box": [1, 2, 3, 4]}],
    )


def test_saved_answers_are_read_back_as_they_were_saved(tmp_path):
    store = SavedAnswers(tmp_path / "var" / "demo-answers.json", {"composer": "baseline"})
    answers = {("doc-1", "How many doors?"): response(), ("doc-2", "How many doors?"): response("Answer: 2")}

    store.save(answers)

    assert store.load() == answers


def test_saved_answers_of_other_settings_are_not_used(tmp_path):
    path = tmp_path / "demo-answers.json"
    SavedAnswers(path, {"composer": "baseline"}).save({("doc-1", "q"): response()})

    assert SavedAnswers(path, {"composer": "combined"}).load() == {}


def test_saved_answers_are_empty_without_a_readable_file(tmp_path):
    store = SavedAnswers(tmp_path / "demo-answers.json", {})

    assert store.load() == {}
    store.path.write_text("not json")
    assert store.load() == {}


def test_demo_answers_command_saves_an_answer_for_each_question(tmp_path, monkeypatch, capsys):
    folder = tmp_path / "data" / "floorplans" / "data" / "swiss_dwellings"
    folder.mkdir(parents=True)
    (folder / "here.png").write_bytes(png())
    file = tmp_path / "set.json"
    file.write_text(json.dumps([
        {"path": "floorplans/data/swiss_dwellings/here.png", "title": "Here", "questions": [
            {"label": "Doors", "text": "How many doors?", "reference": "7"},
            {"label": "Windows", "text": "How many windows?", "reference": "3"},
        ]},
        {"path": "floorplans/data/swiss_dwellings/gone.png", "title": "Gone", "questions": [
            {"label": "Doors", "text": "How many doors?"},
        ]},
    ]))
    monkeypatch.setattr("chatchtg.demo.DEMO_SET", file)
    monkeypatch.setenv("CHATCHTG_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("CHATCHTG_STATE_DIR", str(tmp_path / "state"))

    assert main(["demo-answers"]) == 0

    saved = build_saved_answers(Settings.from_env()).load()
    assert sorted(message for _, message in saved) == ["How many doors?", "How many windows?"]
    assert len({document_id for document_id, _ in saved}) == 1          # the drawing that is there
    assert all(answer.model == "fake" for answer in saved.values())
    out = capsys.readouterr().out
    assert "Here · Doors [fake]" in out and "benchmark: 7" in out and "Answers saved: 2." in out


def test_demo_answers_command_needs_a_drawing_of_the_set(tmp_path, monkeypatch):
    monkeypatch.setenv("CHATCHTG_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("CHATCHTG_STATE_DIR", str(tmp_path / "state"))

    assert main(["demo-answers"]) == 2
    assert not (tmp_path / "state" / "demo-answers.json").exists()
