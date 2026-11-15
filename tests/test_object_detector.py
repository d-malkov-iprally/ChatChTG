import json

import pytest
from PIL import Image

from object_detector.classes import CLASSES, FIXTURES
from object_detector.cubicasa_svg import Box, _path_points, read_boxes
from object_detector.dataset import build_dataset, image_name, label_counts, subset_dataset
from object_detector.scoring import (
    benchmark_answers,
    format_scores,
    score_against_benchmark,
    score_counts,
)

DOOR = """
<g id="Door" class="Door Swing Beside">
  <polygon points="100,200 180,200 180,210 100,210 "/>
  <g id="Threshold"><polygon points="100,200 180,200 180,210 100,210 "/></g>
  <g id="Panel" class="Panel Left Positive">
    <g id="PanelArea"><polygon points="100,200 180,200 180,210 100,210 "/></g>
    <path d="M180,210 q0.00,80 -80,80 l0.00,-80Z"/>
  </g>
</g>"""
OPENING = """
<g id="Door" class="Door None Beside"><polygon points="300,200 380,200 380,210 300,210 "/></g>"""
WINDOW = """
<g id="Window" class="Window Regular"><polygon points="10,20 110,20 110,30 10,30 "/></g>"""
# Turned by 90 degrees and moved: the 40 x 70 boundary becomes 70 wide and 40 high.
TOILET = """
<g class="FixedFurniture Toilet" transform="matrix(0,-1,1,0,500,400)">
  <g class="BoundaryPolygon"><polygon points="0,0 40,0 40,70 0,70"/></g>
  <g class="InnerPolygon"><rect x="0" y="0" width="900" height="900"/></g>
</g>"""
SHOWER_WITH_SCREEN = """
<g class="FixedFurnitureSet" transform="matrix(1,0,0,1,1000,0)">
  <g class="FixedFurniture Shower" transform="matrix(1,0,0,1,10,20)">
    <g class="BoundaryPolygon"><polygon points="0,0 30,0 30,30 0,30"/></g>
  </g>
  <g class="FixedFurniture ShowerScreen" transform="matrix(1,0,0,1,10,20)">
    <g class="BoundaryPolygon"><polygon points="0,0 90,0 90,5 0,5"/></g>
  </g>
</g>"""
SINK_WITHOUT_BOUNDARY = """
<g class="FixedFurniture DoubleSink" transform="matrix(1,0,0,1,600,100)">
  <g><rect x="0" y="0" width="80" height="40"/></g>
</g>"""
HIDDEN_FLOOR = """
<g class="Floor" style="display: none;"><g id="Floor-2" class="Floorplan Floor-2">
  <g id="Window" class="Window Regular"><polygon points="700,20 800,20 800,30 700,30 "/></g>
</g></g>"""


def svg(*parts: str) -> str:
    body = "".join(parts)
    return (
        '<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg"><g id="Model">'
        f'<g class="Floor"><g id="Floor-1" class="Floorplan Floor-1" transform="matrix(1,0,0,1,0,0)">'
        f'<g id="Wall" class="Wall External">{body}</g></g></g></g></svg>'
    )


def test_a_door_box_covers_the_opening_and_the_swing():
    assert read_boxes(svg(DOOR)) == [Box("door", 100, 200, 180, 290)]


def test_an_opening_without_a_door_leaf_is_not_a_door():
    assert read_boxes(svg(OPENING)) == []


def test_a_window_box_is_its_polygon():
    assert read_boxes(svg(WINDOW)) == [Box("window", 10, 20, 110, 30)]


def test_a_fixture_box_is_its_boundary_polygon_after_the_transform():
    assert read_boxes(svg(TOILET)) == [Box("toilet", 500, 360, 570, 400)]


def test_the_transforms_of_all_parents_apply_and_a_shower_screen_is_not_a_shower():
    assert read_boxes(svg(SHOWER_WITH_SCREEN)) == [Box("shower", 1010, 20, 1040, 50)]


def test_a_fixture_without_a_boundary_polygon_takes_the_box_of_its_shapes():
    assert read_boxes(svg(SINK_WITHOUT_BOUNDARY)) == [Box("sink", 600, 100, 680, 140)]


def test_the_floors_that_the_file_hides_are_read_too():
    two_floors = svg(WINDOW).replace("</g></svg>", HIDDEN_FLOOR + "</g></svg>")
    assert [box.left for box in read_boxes(two_floors)] == [10, 700]


def test_every_fixture_maps_to_a_class():
    assert set(FIXTURES.values()) <= set(CLASSES)
    assert "Urinal" not in FIXTURES
    assert "ShowerScreen" not in FIXTURES
    assert "ShowerCab" not in FIXTURES


def make_raw(root, plans: dict[str, str], scaled=(200, 100), original=(100, 50)):
    """A CubiCasa5K folder. ``plans`` is plan -> split file. Every plan has one window."""
    for plan, split_file in plans.items():
        plan_dir = root / plan.strip("/")
        plan_dir.mkdir(parents=True)
        Image.new("RGB", scaled, "white").save(plan_dir / "F1_scaled.png")
        Image.new("RGB", original, "white").save(plan_dir / "F1_original.png")
        (plan_dir / "model.svg").write_text(svg(WINDOW))
        with (root / split_file).open("a") as file:
            file.write(plan + "\n")
    for split_file in ("train.txt", "val.txt", "test.txt"):
        (root / split_file).touch()
    return root


def test_build_dataset_writes_coco_in_the_pixels_of_the_original_image(tmp_path):
    raw = make_raw(tmp_path / "raw", {"/high_quality/7/": "train.txt"})

    stats = build_dataset(raw, tmp_path / "out")

    coco = json.loads((tmp_path / "out/train/_annotations.coco.json").read_text())
    assert coco["images"] == [{"id": 0, "file_name": "high_quality-7.png", "width": 100, "height": 50}]
    assert (tmp_path / "out/train/high_quality-7.png").is_file()
    # The window 10,20-110,30 of the 200 x 100 image is 5,10-55,15 in the 100 x 50 image.
    assert coco["annotations"][0]["bbox"] == [5.0, 10.0, 50.0, 5.0]
    assert coco["annotations"][0]["category_id"] == CLASSES.index("window")
    assert [category["id"] for category in coco["categories"]] == list(range(len(CLASSES)))
    assert stats["splits"]["train"]["boxes"]["window"] == 1
    assert json.loads((tmp_path / "out/valid/_annotations.coco.json").read_text())["images"] == []


def test_build_dataset_skips_a_plan_whose_labels_are_outside_the_image(tmp_path):
    raw = make_raw(tmp_path / "raw", {"/colorful/1/": "test.txt", "/colorful/2/": "test.txt"})
    outside = WINDOW.replace("10,20 110,20 110,30 10,30", "900,20 990,20 990,30 900,30")
    (raw / "colorful/2/model.svg").write_text(svg(outside))

    stats = build_dataset(raw, tmp_path / "out")

    assert stats["splits"]["test"]["images"] == 1
    assert list(stats["splits"]["test"]["skipped_plans"]) == ["/colorful/2/"]
    assert list(label_counts(tmp_path / "out/test")) == ["colorful-1.png"]


def test_subset_dataset_keeps_the_first_images_and_their_labels(tmp_path):
    plans = {f"/colorful/{n}/": "train.txt" for n in range(3)}
    raw = make_raw(tmp_path / "raw", plans)
    build_dataset(raw, tmp_path / "out")

    subset_dataset(tmp_path / "out", tmp_path / "small", limit=2)

    coco = json.loads((tmp_path / "small/train/_annotations.coco.json").read_text())
    assert [image["file_name"] for image in coco["images"]] == ["colorful-0.png", "colorful-1.png"]
    assert {annotation["image_id"] for annotation in coco["annotations"]} == {0, 1}
    assert Image.open(tmp_path / "small/train/colorful-1.png").size == (100, 50)


def test_image_name_is_the_name_of_the_plan_in_the_benchmark():
    assert image_name("/high_quality_architectural/11926/") == "high_quality_architectural-11926.png"


def test_score_counts_gives_the_exact_rate_and_the_mean_error():
    reference = {"a.png": {"door": 2, "window": 1}, "b.png": {"door": 4}}
    predicted = {"a.png": {"door": 2, "window": 3}}

    scores = score_counts(predicted, reference, ["door", "window"])

    assert scores["door"] == pytest.approx(
        {"images": 2, "exact": 1, "exact_rate": 0.5, "mean_abs_error": 2.0, "objects": 6}
    )
    assert scores["window"]["exact"] == 1  # b.png has no window in either
    assert scores["window"]["mean_abs_error"] == 1.0


def test_counts_are_scored_against_the_answers_of_the_benchmark(tmp_path):
    rows = [
        ("data/cubicasa5k/a.png", "How many windows are shown on this floor plan?", 2),
        ("data/cubicasa5k/a.png", "How many showers and bathtubs are drawn in total?", 3),
        ("data/cubicasa5k/a.png", "Kuinka monta ikkunaa pohjapiirustuksessa on?", 2),
        ("data/swiss_dwellings/s.png", "How many windows are shown on this floor plan?", 9),
    ]
    questions = tmp_path / "questions.jsonl"
    questions.write_text(
        "\n".join(json.dumps({"path": p, "question": q, "answer": a}) for p, q, a in rows)
    )

    answers = benchmark_answers(questions)
    scores = score_against_benchmark({"a.png": {"window": 2, "shower": 1, "bathtub": 1}}, answers)

    assert list(answers) == ["a.png"]
    assert scores["How many windows are shown"]["exact_rate"] == 1.0
    assert scores["How many showers and bathtubs"]["mean_abs_error"] == 1.0
    assert scores["How many doors are shown"]["images"] == 0
    assert "100.0%" in format_scores(scores)


def test_a_path_with_numbers_after_a_close_command_is_not_read():
    # It once never returned: the numbers after Z were never consumed.
    assert _path_points("M0,0 L10,0 Z 5,5") == []
    assert _path_points("M0,0 L10,0 Z M5,5 L6,6") != []
