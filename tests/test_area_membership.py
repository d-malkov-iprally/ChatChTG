import cv2
import numpy as np

from chatchtg.areas.membership import assign_apartments, outdoor_rooms

RED = (176, 58, 46)
OUTDOOR = (228, 228, 228)


def dash_dot(image, start, end):
    """A red dash-dot line as on the plans: 14 px dash, 5 px gap, 3 px dot, 5 px gap."""
    start, end = np.array(start, float), np.array(end, float)
    length = float(np.linalg.norm(end - start))
    direction = (end - start) / length
    at = 0.0
    while at < length:
        for stroke in (14, 3):
            a = start + direction * at
            b = start + direction * min(at + stroke, length)
            cv2.line(image, tuple(int(v) for v in a), tuple(int(v) for v in b), RED, 2)
            at += stroke + 5
            if at >= length:
                break


def loop(image, x0, y0, x1, y1):
    for start, end in (
        ((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0)),
    ):  # fmt: skip
        dash_dot(image, start, end)


def plan():
    """Two apartments side by side, a stair between them and a balcony below the first."""
    image = np.full((400, 700, 3), 255, np.uint8)
    loop(image, 40, 40, 300, 240)
    loop(image, 400, 40, 660, 240)
    cv2.rectangle(image, (60, 260), (200, 340), OUTDOOR, -1)
    cv2.rectangle(image, (60, 260), (200, 340), (60, 60, 60), 2)
    return image


def test_rooms_get_the_apartment_whose_outline_holds_them():
    rooms = [(100, 100), (250, 200), (500, 120), (350, 150), (130, 300)]
    tags = {"A": (170, 70), "B": (530, 70)}
    assert assign_apartments(plan(), rooms, tags) == ["A", "A", "B", None, None]


def test_point_on_the_outline_or_on_a_glyph_still_gets_its_apartment():
    image = plan()
    cv2.putText(image, "1.01", (80, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (90, 90, 90), 2)
    rooms = [(100, 104), (43, 150), (657, 150)]
    assert assign_apartments(image, rooms, {"A": (170, 70), "B": (530, 70)}) == ["A", "A", "B"]


def test_tag_box_drawn_over_the_outline_does_not_open_the_apartment():
    image = plan()
    cv2.rectangle(image, (135, 28), (205, 52), (255, 255, 255), -1)
    cv2.rectangle(image, (135, 28), (205, 52), (130, 130, 130), 1)
    rooms = [(100, 100), (500, 120), (350, 200)]
    tags = {"A": (170, 40), "B": (530, 70)}
    assert assign_apartments(image, rooms, tags) == ["A", "B", None]


def test_no_tags_means_no_apartments():
    assert assign_apartments(plan(), [(100, 100)], {}) == [None]
    assert assign_apartments(plan(), [], {"A": (170, 70)}) == []


def test_room_with_a_grey_floor_is_outdoor():
    image = plan()
    cv2.putText(image, "1.09", (100, 305), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (90, 90, 90), 2)
    assert outdoor_rooms(image, [(130, 300), (124, 296), (100, 100), (500, 120)]) == [
        True, True, False, False,
    ]  # fmt: skip


def test_point_outside_the_image_is_not_outdoor():
    assert outdoor_rooms(plan(), [(-50, -50), (5000, 5000)]) == [False, False]
    assert assign_apartments(plan(), [(-50, -50), (5000, 5000)], {"A": (170, 70)}) == [None, None]
