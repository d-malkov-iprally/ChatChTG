import json

import pytest

from chatchtg.cli import main
from chatchtg.evals.benchmark import build_benchmark, sample_subset, select_questions
from conftest import PNG


def _question(path, question="Windows?", answer=4, unit="count", question_type="count"):
    return {"path": path, "question": question, "answer": answer, "unit": unit, "type": question_type}


def _write_questions(dataset_dir, questions):
    (dataset_dir / "questions.jsonl").write_text(
        "\n".join(json.dumps(q) for q in questions) + "\n", encoding="utf-8"
    )


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.fixture
def full(data_root):
    """A small copy of the full dataset: two floor-plan sources and one other source."""
    dataset_dir = data_root / "full"
    files = {
        "cubicasa5k/a.png": PNG + b"a",
        "cubicasa5k/b.png": PNG + b"b",
        "cubicasa5k/unasked.png": PNG,
        "swiss_dwellings/s1.pdf": b"%PDF-1.7",
        "swiss_dwellings/s1.png": PNG + b"s1",
        "swiss_dwellings/s1.dxf": b"0\nSECTION",
        "swiss_dwellings/no_png.pdf": b"%PDF-1.7",
        "ifc_models/m_plan.pdf": b"%PDF-1.7",
        "ifc_models/m_plan.png": PNG,
        "finna/scan.jpg": b"\xff\xd8\xff",
    }
    for name, content in files.items():
        target = dataset_dir / "data" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    for source in ("cubicasa5k", "swiss_dwellings", "ifc_models"):
        (dataset_dir / "data" / source / "LICENSE.md").write_text(f"licence of {source}")
    _write_questions(
        dataset_dir,
        [
            _question("data/cubicasa5k/a.png"),
            _question("data/cubicasa5k/a.png", "Kuinka monta ikkunaa?"),
            _question("data/cubicasa5k/b.png", "Doors?", 2),
            _question("data/swiss_dwellings/s1.pdf", "Apartments?", 2),
            _question("data/swiss_dwellings/s1.pdf", "Fläche von Raum 1?", 4.17, "m2", "area"),
            _question("data/swiss_dwellings/no_png.pdf"),
            _question("data/ifc_models/m_plan.pdf", "Area?", 2.63, "m2", "area"),
        ],
    )
    return dataset_dir


def test_select_keeps_only_floor_plan_sources_and_points_at_the_png(full):
    questions, dropped = select_questions(full)

    assert [q["path"] for q in questions] == [
        "data/cubicasa5k/a.png",
        "data/cubicasa5k/a.png",
        "data/cubicasa5k/b.png",
        "data/swiss_dwellings/s1.png",
        "data/swiss_dwellings/s1.png",
    ]
    assert dropped == 1  # no_png.pdf has no raster copy
    assert questions[4] == _question("data/swiss_dwellings/s1.png", "Fläche von Raum 1?", 4.17, "m2", "area")


@pytest.mark.parametrize(
    "escape",
    ["data/cubicasa5k/../../outside.png", "data/cubicasa5k/../ifc_models/m_plan.png", "{absolute}"],
)
def test_select_refuses_paths_that_leave_the_source_folder(full, escape):
    outside = full.parent / "outside.png"
    outside.write_bytes(PNG)
    _write_questions(full, [_question(escape.format(absolute=outside))])

    assert select_questions(full) == ([], 0)


def test_select_drops_a_question_whose_image_is_missing(full):
    _write_questions(full, [_question("data/cubicasa5k/gone.png")])

    assert select_questions(full) == ([], 1)


def test_build_writes_a_self_contained_dataset(full, data_root):
    out_dir = data_root / "floorplans"

    report = build_benchmark(full, out_dir, subset_size=2)

    copied = sorted(p.relative_to(out_dir).as_posix() for p in (out_dir / "data").rglob("*") if p.is_file())
    assert copied == [
        "data/cubicasa5k/LICENSE.md",
        "data/cubicasa5k/a.png",
        "data/cubicasa5k/b.png",
        "data/swiss_dwellings/LICENSE.md",
        "data/swiss_dwellings/s1.png",
    ]
    assert (out_dir / "data/cubicasa5k/a.png").read_bytes() == PNG + b"a"
    rows = _read_jsonl(out_dir / "questions.jsonl")
    assert len(rows) == 5
    assert all(set(row) == {"path", "question", "answer", "unit", "type"} for row in rows)
    assert all((out_dir / row["path"]).is_file() for row in rows)
    assert "Fläche" in (out_dir / "questions.jsonl").read_text(encoding="utf-8")
    assert len(_read_jsonl(out_dir / "questions-subset.jsonl")) == 2
    assert report["questions"] == 5
    assert report["images"] == 3
    assert report["dropped"] == 1
    assert report["subset"] == 2
    assert report["copied"] == 3
    assert report["by_source"] == {
        "cubicasa5k": {"questions": 3, "images": 2},
        "swiss_dwellings": {"questions": 2, "images": 1},
    }
    assert report["by_type"] == {"area": 1, "count": 4}
    readme = (out_dir / "README.md").read_text(encoding="utf-8")
    assert "cubicasa5k" in readme and "LICENSE.md" in readme


def test_build_twice_copies_nothing_and_writes_the_same_files(full, data_root):
    out_dir = data_root / "floorplans"
    build_benchmark(full, out_dir, subset_size=2)
    names = ("questions.jsonl", "questions-subset.jsonl", "README.md")
    first = [(out_dir / name).read_bytes() for name in names]

    report = build_benchmark(full, out_dir, subset_size=2)

    assert report["copied"] == 0
    assert [(out_dir / name).read_bytes() for name in names] == first


def test_build_refuses_to_write_into_the_source_dataset(full):
    with pytest.raises(ValueError, match="source dataset"):
        build_benchmark(full, full)


def _many_questions():
    questions = []
    for drawing in range(20):
        for number in range(5):
            questions.append(_question(f"data/cubicasa5k/{drawing}.png", f"Count {number}?"))
            questions.append(_question(f"data/swiss_dwellings/{drawing}.png", f"Count {number}?"))
            questions.append(
                _question(f"data/swiss_dwellings/{drawing}.png", f"Area {number}?", 1.5, "m2", "area")
            )
    return questions


def test_subset_is_fixed_by_the_seed():
    questions = _many_questions()

    assert sample_subset(questions, 30, seed=0) == sample_subset(questions, 30, seed=0)
    assert sample_subset(questions, 30, seed=0) != sample_subset(questions, 30, seed=1)


def test_subset_is_balanced_over_source_and_type_and_spreads_over_drawings():
    subset = sample_subset(_many_questions(), 30, seed=0)

    assert len(subset) == 30
    strata = [(q["path"].split("/")[1], q["type"]) for q in subset]
    assert {stratum: strata.count(stratum) for stratum in set(strata)} == {
        ("cubicasa5k", "count"): 10,
        ("swiss_dwellings", "count"): 10,
        ("swiss_dwellings", "area"): 10,
    }
    cubicasa = [q["path"] for q in subset if "cubicasa5k" in q["path"]]
    assert len(set(cubicasa)) == 10  # one question per drawing while drawings remain


def test_subset_keeps_the_order_of_the_full_set():
    questions = _many_questions()
    subset = sample_subset(questions, 30, seed=0)

    positions = [questions.index(q) for q in subset]
    assert positions == sorted(positions)


def test_subset_is_everything_when_the_set_is_smaller_than_the_size():
    questions = _many_questions()[:7]

    assert sample_subset(questions, 500, seed=0) == questions


def test_subset_fills_up_from_larger_strata_when_one_is_small():
    questions = [q for q in _many_questions() if q["type"] == "count"]
    questions.append(_question("data/swiss_dwellings/0.png", "Area?", 1.5, "m2", "area"))

    subset = sample_subset(questions, 31, seed=0)

    assert len(subset) == 31
    assert sum(q["type"] == "area" for q in subset) == 1


def test_cli_builds_the_benchmark_and_eval_reads_the_subset(full, data_root, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CHATCHTG_DATA_ROOT", str(data_root))
    monkeypatch.setenv("CHATCHTG_STATE_DIR", str(tmp_path / "state"))
    out_dir = data_root / "floorplans"

    assert main(["benchmark", str(full), "--subset-size", "2"]) == 0
    printed = capsys.readouterr().out
    assert "Questions: 5" in printed and "Images: 3" in printed
    assert (out_dir / "questions.jsonl").is_file()

    result = tmp_path / "result.jsonl"
    assert main(["eval", str(out_dir), "--questions", "questions-subset.jsonl", "--out", str(result)]) == 0
    assert len(result.read_text().splitlines()) == 2
    assert main(["eval", str(out_dir), "--out", str(result)]) == 0
    assert len(result.read_text().splitlines()) == 5


def test_cli_benchmark_without_questions_file_writes_nothing(data_root, monkeypatch, capsys):
    monkeypatch.setenv("CHATCHTG_DATA_ROOT", str(data_root))

    assert main(["benchmark", str(data_root / "missing")]) == 2
    assert "No questions.jsonl" in capsys.readouterr().err
    assert not (data_root / "floorplans").exists()
