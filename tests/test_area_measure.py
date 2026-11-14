"""The room measurer on small synthetic plans drawn in the style of the benchmark."""

import cv2
import numpy as np
import pytest

from chatchtg.areas.measure import measure_rooms

PPM = 59.055  # 1:100 at 150 dpi
FINE = 8  # the plan is drawn 8 times larger and averaged down, which anti-aliases it


class Sketch:
    def __init__(self, width: int = 420, height: int = 320) -> None:
        self.canvas = np.full((height * FINE, width * FINE), 255, np.uint8)

    def _box(self, x0, y0, x1, y1, grey) -> None:
        points = np.round(np.array([x0, y0, x1, y1]) * FINE).astype(int)
        cv2.rectangle(self.canvas, (points[0], points[1]), (points[2] - 1, points[3] - 1), int(grey), -1)

    def wall(self, x0, y0, x1, y1) -> None:
        """A wall: grey fill with a black outline 2 px wide, centred on its edge."""
        self._box(x0 - 1, y0 - 1, x1 + 1, y1 + 1, 0)
        self._box(x0 + 1, y0 + 1, x1 - 1, y1 - 1, 176)

    def line(self, x0, y0, x1, y1, grey, width) -> None:
        """A horizontal or vertical line centred on the given segment."""
        half = width / 2
        if x0 == x1:
            self._box(x0 - half, y0, x0 + half, y1, grey)
        else:
            self._box(x0, y0 - half, x1, y0 + half, grey)

    def image(self) -> np.ndarray:
        height, width = self.canvas.shape
        small = cv2.resize(self.canvas, (width // FINE, height // FINE), interpolation=cv2.INTER_AREA)
        return np.stack([small] * 3, axis=2)


def room_with_walls(x0=60.3, y0=50.6, x1=300.4, y1=230.2, door=None) -> Sketch:
    """One room whose inner wall faces are at the given coordinates. Walls are 12 px thick."""
    sketch = Sketch()
    sketch.wall(x0 - 12, y0 - 12, x1 + 12, y0)  # top
    sketch.wall(x0 - 12, y1, x1 + 12, y1 + 12)  # bottom
    sketch.wall(x0 - 12, y0 - 12, x0, y1 + 12)  # left
    if door is None:
        sketch.wall(x1, y0 - 12, x1 + 12, y1 + 12)  # right
    else:
        top, bottom = door
        sketch.wall(x1, y0 - 12, x1 + 12, top)
        sketch.wall(x1, bottom, x1 + 12, y1 + 12)
        sketch.line(x1, top, x1, bottom, 90, 1.0)  # the two faces of the opening
        sketch.line(x1 + 12, top, x1 + 12, bottom, 90, 1.0)
    return sketch


def area(x0, y0, x1, y1) -> float:
    return (x1 - x0) * (y1 - y0) / PPM**2


def test_a_plain_room_is_measured_to_the_centre_of_the_wall_outline():
    sketch = room_with_walls()
    [room] = measure_rooms(sketch.image(), [(180, 140)], PPM)
    assert room.area_m2 == pytest.approx(area(60.3, 50.6, 300.4, 230.2), rel=0.005)
    assert not room.flags


def test_a_fixture_drawn_in_the_room_does_not_cut_it():
    sketch = room_with_walls()
    sketch.line(120, 50.6, 120, 230.2, 58, 1.0)  # a counter edge from wall to wall
    [room] = measure_rooms(sketch.image(), [(200, 140)], PPM)
    assert room.area_m2 == pytest.approx(area(60.3, 50.6, 300.4, 230.2), rel=0.005)


def test_a_light_grey_separator_divides_two_rooms():
    sketch = room_with_walls()
    sketch.line(170.5, 50.6, 170.5, 230.2, 160, 1.6)
    left, right = measure_rooms(sketch.image(), [(110, 140), (240, 140)], PPM)
    assert left.area_m2 == pytest.approx(area(60.3, 50.6, 170.5, 230.2), rel=0.01)
    assert right.area_m2 == pytest.approx(area(170.5, 50.6, 300.4, 230.2), rel=0.01)


def test_a_door_opening_belongs_to_no_room():
    sketch = room_with_walls(door=(100.0, 153.0))
    [room] = measure_rooms(sketch.image(), [(180, 140)], PPM)
    assert room.area_m2 == pytest.approx(area(60.3, 50.6, 300.4, 230.2), rel=0.01)


def test_a_seed_outside_any_room_is_flagged():
    sketch = room_with_walls()
    [room] = measure_rooms(sketch.image(), [(-5, -5)], PPM)
    assert room.area_m2 is None
    assert "no-region" in room.flags


def test_an_image_without_free_pixels_has_no_rooms():
    rooms = measure_rooms(np.zeros((300, 300, 3), np.uint8), [(10, 10)], 59.0)

    assert rooms[0].area_m2 is None
    assert rooms[0].flags == ["no-region"]


def test_debug_receives_the_component_owners():
    image = room_with_walls().image()
    debug: dict = {}
    [room] = measure_rooms(image, [(180, 140)], PPM, debug=debug)
    [plain] = measure_rooms(image, [(180, 140)], PPM)

    assert set(debug) == {"labels", "stats", "owner", "neutral", "graph", "exterior", "raster", "seed_of"}
    assert debug["owner"][debug["seed_of"][0]] == 0
    assert room.area_m2 == plain.area_m2
