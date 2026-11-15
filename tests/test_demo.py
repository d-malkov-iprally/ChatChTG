import io
import json

from PIL import Image

from chatchtg.demo import DemoDrawing, DemoQuestion, load_demo_set, register_demo_set


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
