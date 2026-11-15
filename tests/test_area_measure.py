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


PPM_200 = 29.5276  # 1:200 at 150 dpi


def area_200(x0, y0, x1, y1) -> float:
    return (x1 - x0) * (y1 - y0) / PPM_200**2


def bathroom_at_1_200(x0=60.0, y0=50.0, x1=116.0, y1=110.0, tub=26.0) -> Sketch:
    """A bathroom with a tub along its bottom wall, 26 px (0.88 m) deep.

    The tub is drawn as on site1273_bldg2474_floor11569, room 1.47: it stands 2 px off the
    wall outlines and its rim is two black strokes about 1 px wide and 2.4 px apart. At
    1:200 the strokes leave no free pixel between them, so the rim is a band of ink four
    pixels wide.
    """
    sketch = room_with_walls(x0, y0, x1, y1)
    for inset in (0.0, 2.4):
        left, top, right, bottom = x0 + 2 + inset, y1 - tub + inset, x1 - 2 - inset, y1 - 2 - inset
        sketch.line(left, top, right, top, 0, 1.2)
        sketch.line(left, bottom, right, bottom, 0, 1.2)
        sketch.line(left, top, left, bottom, 0, 1.2)
        sketch.line(right, top, right, bottom, 0, 1.2)
    return sketch


def test_a_bathtub_with_a_thick_outline_stays_in_its_room_at_1_200():
    sketch = bathroom_at_1_200()
    [room] = measure_rooms(sketch.image(), [(88, 66)], PPM_200)
    assert room.area_m2 == pytest.approx(area_200(60, 50, 116, 110), rel=0.02)


def owned(debug: dict, x: int, y: int) -> bool:
    """Whether the free component at a pixel was joined to a room."""
    return int(debug["labels"][y, x]) in debug["owner"]


def test_a_small_room_without_a_label_is_not_joined_to_its_neighbour():
    """A room 1.0 m wide whose label was not read, behind a wall too thin to show its fill."""
    sketch = room_with_walls(60, 50, 134, 110)
    sketch.line(102, 50, 102, 110, 0, 4.0)
    debug: dict = {}
    [room] = measure_rooms(sketch.image(), [(80, 80)], PPM_200, debug=debug)
    # The outline of a room leaves out a piece behind a wall, so the area alone does not
    # show a wrong join: the owner of the neighbour's free space does.
    assert owned(debug, 80, 80) and not owned(debug, 118, 80)
    assert room.area_m2 == pytest.approx(area_200(60, 50, 101, 110), rel=0.02)


def test_a_shaft_behind_a_wall_is_not_joined():
    """A shaft 20 x 30 px in a block of wall, 4 px behind the wall face of the room: within
    reach of the room's free space, so only the wall fill around it keeps it out."""
    sketch = room_with_walls(60, 50, 120, 110)
    sketch.wall(120, 38, 164, 122)
    sketch._box(124, 61, 144, 91, 255)
    sketch.line(123, 60, 145, 60, 0, 2)
    sketch.line(123, 92, 145, 92, 0, 2)
    sketch.line(123, 60, 123, 92, 0, 2)
    sketch.line(145, 60, 145, 92, 0, 2)
    debug: dict = {}
    [room] = measure_rooms(sketch.image(), [(90, 80)], PPM_200, debug=debug)
    assert owned(debug, 90, 80) and not owned(debug, 134, 76)
    assert room.area_m2 == pytest.approx(area_200(60, 50, 120, 110), rel=0.02)


def test_a_neighbouring_room_is_not_joined_through_a_thin_wall():
    sketch = room_with_walls(60, 50, 150, 110)
    sketch.line(100, 50, 100, 110, 0, 4.0)  # a wall too thin to show its fill
    left, right = measure_rooms(sketch.image(), [(80, 80), (125, 80)], PPM_200)
    assert left.area_m2 == pytest.approx(area_200(60, 50, 99, 110), rel=0.02)
    assert right.area_m2 == pytest.approx(area_200(101, 50, 150, 110), rel=0.02)


def test_a_fixture_line_along_a_wall_does_not_move_the_edge():
    sketch = room_with_walls()
    sketch.line(62.8, 50.6, 62.8, 230.2, 0, 1.3)  # 2.5 px inside the left wall face
    [room] = measure_rooms(sketch.image(), [(180, 140)], PPM)
    assert room.area_m2 == pytest.approx(area(60.3, 50.6, 300.4, 230.2), rel=0.004)


def test_a_fixture_line_along_a_wall_at_1_200():
    sketch = room_with_walls(60, 50, 120, 100)
    sketch.line(62.5, 50, 62.5, 100, 0, 1.3)
    [room] = measure_rooms(sketch.image(), [(90, 75)], PPM_200)
    assert room.area_m2 == pytest.approx(area_200(60, 50, 120, 100), rel=0.01)


def thin_left_wall(fixture: bool) -> Sketch:
    """A room whose left wall is a black line 4 px thick, without fill."""
    sketch = Sketch()
    sketch.wall(48, 38, 312, 50)  # top
    sketch.wall(48, 230, 312, 242)  # bottom
    sketch.wall(300, 38, 312, 242)  # right
    sketch.line(58, 38, 58, 242, 0, 4.0)
    if fixture:
        sketch.line(61.5, 50, 61.5, 230, 0, 1.3)  # it touches the wall
    return sketch


def test_a_fixture_line_along_a_thin_black_wall_keeps_the_edge_out_of_the_wall():
    [plain] = measure_rooms(thin_left_wall(fixture=False).image(), [(180, 140)], PPM)
    [room] = measure_rooms(thin_left_wall(fixture=True).image(), [(180, 140)], PPM)
    assert room.area_m2 <= plain.area_m2 * 1.01
    assert room.area_m2 >= plain.area_m2 * 0.97
