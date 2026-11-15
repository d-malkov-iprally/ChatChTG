"""The Swiss labels from DXF, the split and the tiles. No data, no network, no Modal."""

import math
from collections import Counter

import ezdxf
import numpy as np
import pytest

from object_detector.cubicasa_svg import Box
from object_detector.swiss_dataset import (
    STRIDE,
    TILE,
    boxes_in_tile,
    check_labels,
    drawing_area,
    join_tiles,
    score_plans,
    site_of,
    split_of,
    tiles,
)
from object_detector.swiss_dxf import (
    Transform,
    boxes_in_pixels,
    fit_transform,
    pixels_per_metre,
    read_drawing,
    read_symbols,
)


def rect(x0, y0, x1, y1):
    return [(x1, y0), (x0, y0), (x0, y1), (x1, y1)]


def bowl(cx, cy, rx, ry, n=48):
    return [(cx + rx * math.cos(t), cy + ry * math.sin(t)) for t in np.linspace(0, math.pi, n)]


def synthetic_dxf(path):
    """A small plan in the style of the Swiss renderer, one or two of each object."""
    doc = ezdxf.new()
    m = doc.modelspace()
    for name in ("DOOR", "WINDOW", "FIXTURE", "WALL", "ANNOTATION"):
        doc.layers.add(name)
    m.add_text("Massstab 1:200 (A3)", dxfattribs={"layer": "ANNOTATION"})
    m.add_lwpolyline(rect(-5, -5, 5, 5), close=True, dxfattribs={"layer": "WALL"})
    # A single door: an opening (5 points, open), its leaf and its arc on the hinge corner.
    m.add_lwpolyline(rect(0, 0, 0.9, 0.1) + [(0.9, 0)], dxfattribs={"layer": "DOOR"})
    m.add_lwpolyline([(0, 0.1), (0, 1.0)], dxfattribs={"layer": "DOOR"})
    m.add_arc((0, 0.1), 0.9, 0, 90, dxfattribs={"layer": "DOOR"})
    # A double door: one opening, two arcs. It is one door.
    m.add_lwpolyline(rect(2, 0, 3.6, 0.1) + [(3.6, 0)], dxfattribs={"layer": "DOOR"})
    m.add_arc((2, 0.1), 0.8, 0, 90, dxfattribs={"layer": "DOOR"})
    m.add_arc((3.6, 0.1), 0.8, 90, 180, dxfattribs={"layer": "DOOR"})
    # A window: an outline and two glass lines.
    m.add_lwpolyline(rect(-4, 4.9, -2.5, 5.0), close=True, dxfattribs={"layer": "WINDOW"})
    m.add_lwpolyline([(-4, 4.93), (-2.5, 4.93)], dxfattribs={"layer": "WINDOW"})
    m.add_lwpolyline([(-4, 4.97), (-2.5, 4.97)], dxfattribs={"layer": "WINDOW"})
    fixture = {"layer": "FIXTURE"}
    # Toilet: outline, bowl, tank line.
    m.add_lwpolyline(rect(-4, -4, -3.6, -3.4), close=True, dxfattribs=fixture)
    m.add_lwpolyline(bowl(-3.8, -3.75, 0.15, 0.25), dxfattribs=fixture)
    m.add_lwpolyline([(-3.95, -3.5), (-3.65, -3.5)], dxfattribs=fixture)
    # Washbasin: outline, bowl, drain.
    m.add_lwpolyline(rect(-3, -4, -2.5, -3.6), close=True, dxfattribs=fixture)
    m.add_lwpolyline(bowl(-2.75, -3.85, 0.2, 0.12), dxfattribs=fixture)
    m.add_circle((-2.75, -3.8), 0.03, dxfattribs=fixture)
    # Bathtub: outline, 69-point inner line, drain.
    m.add_lwpolyline(rect(0, -4, 0.7, -2.3), close=True, dxfattribs=fixture)
    m.add_lwpolyline(bowl(0.35, -3.2, 0.3, 0.8, n=69), dxfattribs=fixture)
    m.add_circle((0.35, -3.9), 0.025, dxfattribs=fixture)
    # A narrow bathtub with a drain only.
    m.add_lwpolyline(rect(1, -4, 1.2, -2.3), close=True, dxfattribs=fixture)
    m.add_circle((1.1, -3.9), 0.025, dxfattribs=fixture)
    # Shower: outline and the 5-point tray mark.
    m.add_lwpolyline(rect(2, -4, 2.8, -3.2), close=True, dxfattribs=fixture)
    m.add_lwpolyline(rect(2.05, -3.95, 2.75, -3.25) + [(2.75, -3.95)], dxfattribs=fixture)
    m.add_circle((2.4, -3.6), 0.045, dxfattribs=fixture)
    # Washing machine: outline and drum.
    m.add_lwpolyline(rect(3.5, -4, 4.1, -3.4), close=True, dxfattribs=fixture)
    m.add_circle((3.8, -3.7), 0.21, dxfattribs=fixture)
    # Kitchen hob: four rings. Not a class.
    m.add_lwpolyline(rect(-1, 3, -0.4, 3.6), close=True, dxfattribs=fixture)
    for x in (-0.85, -0.55):
        for y in (3.15, 3.45):
            m.add_circle((x, y), 0.08, dxfattribs=fixture)
    doc.saveas(path)
    return path


@pytest.fixture
def symbols(tmp_path):
    drawing = read_drawing(synthetic_dxf(tmp_path / "plan.dxf"))
    return drawing, read_symbols(drawing)


def test_the_scale_is_read_from_the_title(symbols):
    drawing, _ = symbols
    assert drawing.scale == 200


def test_each_object_is_one_box_and_a_double_door_is_one_door(symbols):
    _, boxes = symbols
    assert Counter(b.label for b in boxes) == {
        "door": 2, "window": 1, "toilet": 1, "washbasin": 1, "bathtub": 2, "shower": 1,
        "washing machine": 1,
    }


def test_a_door_box_covers_the_opening_and_the_swing(symbols):
    _, boxes = symbols
    doors = sorted((b for b in boxes if b.label == "door"), key=lambda b: b.left)
    assert doors[0].left == pytest.approx(0) and doors[0].right == pytest.approx(0.9)
    assert doors[0].top == pytest.approx(0) and doors[0].bottom == pytest.approx(1.0)
    assert doors[1].left == pytest.approx(2) and doors[1].right == pytest.approx(3.6)
    assert doors[1].bottom == pytest.approx(0.9)


def test_a_fixture_box_is_its_outline(symbols):
    _, boxes = symbols
    (toilet,) = [b for b in boxes if b.label == "toilet"]
    assert (toilet.left, toilet.top, toilet.right, toilet.bottom) == pytest.approx((-4, -4, -3.6, -3.4))
    narrow = [b for b in boxes if b.label == "bathtub" and b.left > 0.9]
    assert narrow and narrow[0].right == pytest.approx(1.2)


def test_metres_map_to_pixels_with_y_down():
    transform = Transform(10.0, 100.0, 200.0)
    (box,) = boxes_in_pixels([Box("door", 1, 2, 3, 5)], transform, (1000, 1000))
    assert (box.left, box.top, box.right, box.bottom) == (110, 150, 130, 180)


def test_the_offset_is_fitted_to_the_walls(tmp_path):
    drawing = read_drawing(synthetic_dxf(tmp_path / "plan.dxf"))
    scale = pixels_per_metre(drawing.scale)
    image = np.full((600, 800, 3), 255, np.uint8)
    # Draw the 10 x 10 m wall square with its corner (-5, 5) at pixel (100, 150).
    side = round(10 * scale)
    image[150:150 + side, 100:100 + side] = 0
    transform, fit = fit_transform(image, drawing)
    assert transform.dx == pytest.approx(100 + 5 * scale, abs=1)
    assert transform.dy == pytest.approx(150 + 5 * scale, abs=1)
    assert fit > 0.95


def test_all_floors_of_a_site_share_a_split_and_the_split_is_stable():
    assert site_of("site1000_bldg2082_floor9385") == "1000"
    assert split_of("site7_bldg1_floor1") == split_of("site7_bldg2_floor9")
    splits = Counter(split_of(f"site{i}_bldg1_floor1") for i in range(3000))
    assert set(splits) == {"train", "valid", "test"}
    assert 0.66 < splits["train"] / 3000 < 0.74
    assert 0.12 < splits["test"] / 3000 < 0.18
    with pytest.raises(ValueError):
        site_of("colorful-1.png")


def test_the_drawing_area_is_inside_the_frame_and_above_the_title_block():
    grey = np.full((1000, 1400), 255, np.uint8)
    grey[30:32, 20:1380] = 0  # top of the frame
    grey[968:970, 20:1380] = 0  # bottom of the frame
    grey[800:802, 20:1380] = 0  # top of the title block
    grey[30:970, 20:22] = 0
    grey[30:970, 1378:1380] = 0
    grey[600:602, 40:1300] = 0  # a long wall in the drawing is not the title block
    assert drawing_area(grey) == (22, 32, 1378, 800)


def test_tiles_cover_the_area_and_end_on_its_edge():
    corners = tiles((0, 0, 1500, 700))
    assert corners[0] == (0, 0)
    assert max(x for x, _ in corners) + TILE == 1500
    assert max(y for _, y in corners) + TILE == 700
    assert tiles((0, 0, 300, 300)) == [(0, 0)]


def test_a_box_cut_by_a_tile_is_kept_only_if_mostly_inside():
    boxes = [Box("door", 600, 10, 700, 50), Box("window", 620, 100, 660, 120)]
    # 40% of the door and 50% of the window are in the first tile.
    assert boxes_in_tile(boxes, 0, 0) == [Box("window", 620, 100, 640, 120)]
    assert boxes_in_tile(boxes, 448, 0) == [
        Box("door", 152, 10, 252, 50), Box("window", 172, 100, 212, 120)
    ]


def test_joined_tiles_count_an_object_in_the_overlap_once():
    area = (0, 0, TILE + STRIDE, TILE)
    seen = {"label": "toilet", "score": 0.9}
    # One toilet at x 500..520 of the sheet, seen by both tiles.
    found = [
        ((0, 0), [{**seen, "box": [500, 10, 520, 30]}]),
        ((STRIDE, 0), [{**seen, "box": [500 - STRIDE, 10, 520 - STRIDE, 30]}]),
    ]
    joined = join_tiles(found, area)
    assert len(joined) == 1 and joined[0]["box"] == [500, 10, 520, 30]


def test_label_check_and_plan_scores_compare_with_the_benchmark():
    labels = {
        "site1_bldg1_floor1": {"scale": 100, "boxes": [["door", 0, 0, 1, 1], ["door", 2, 2, 3, 3]]},
        "site2_bldg1_floor1": {"scale": 200, "boxes": [["shower", 0, 0, 1, 1]]},
    }
    answers = {"site1_bldg1_floor1": {"door": 2}, "site2_bldg1_floor1": {"door": 1, "shower": 1}}
    check = check_labels(labels, answers)
    assert check["door"]["exact"] == 1 and check["door"]["plans"] == 2
    assert check["shower"]["unasked_with_label"] == 0
    scores = score_plans({"site1_bldg1_floor1": {"door": 2}}, labels, answers, list(labels))
    assert scores["door"]["all"]["exact"] == 1
    assert scores["door"]["200"]["mean_abs_error"] == 1
    assert scores["shower"]["100"]["plans"] == 0
