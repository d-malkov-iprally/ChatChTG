import io

from PIL import Image

from chatchtg.evidence import (
    colour_key, describe_method, format_value, geometry, legend_rows, render_evidence,
)

BLUE, ORANGE, VIOLET, RED = (31, 119, 180), (245, 130, 49), (145, 30, 180), (230, 25, 75)


def png(size=(400, 300), mode="RGB", colour="white"):
    buffer = io.BytesIO()
    Image.new(mode, size, colour).save(buffer, format="PNG")
    return buffer.getvalue()


def test_a_box_item_is_drawn_and_the_original_size_is_kept():
    image = render_evidence(png(), [{"text": "MH", "box": [100, 100, 200, 180]}])

    assert image.size == (400, 300) and image.mode == "RGB"
    assert image.getpixel((100, 140)) == RED          # left edge of the box
    assert image.getpixel((300, 250)) == (255, 255, 255)


def test_a_polygon_item_is_outlined_in_the_colour_of_its_kind_and_lightly_filled():
    square = [[50, 100], [150, 100], [150, 200], [50, 200]]

    printed = render_evidence(png(), [{"text": "1.01", "kind": "printed", "polygon": square}])
    measured = render_evidence(png(), [{"text": "1.01", "kind": "measured", "polygon": square}])

    assert printed.getpixel((50, 150)) == BLUE and measured.getpixel((50, 150)) == ORANGE
    inside = printed.getpixel((100, 180))
    assert inside != (255, 255, 255) and inside != BLUE and min(inside) > 180   # a light tint


def test_a_point_item_is_drawn_as_a_ring():
    image = render_evidence(png(), [{"text": "door", "point": [200, 150]}])

    assert image.getpixel((200, 150)) == (255, 255, 255)        # the centre stays readable
    assert any(image.getpixel((x, 150)) == RED for x in range(201, 230))


def test_items_without_or_with_broken_geometry_are_not_drawn_and_do_not_fail():
    evidence = [
        {"text": "Total", "kind": "total", "value": 20.0, "unit": "m²"},
        {"text": "bad", "box": [1, 2, 3]},
        {"text": "bad", "polygon": [[1, 2], [3, 4]]},
        {"text": "bad", "point": ["a", None]},
        {"text": "outside", "box": [-50, -50, 5000, 5000]},
    ]

    image = render_evidence(png(), evidence)

    assert image.size == (400, 300)
    assert [geometry(item) is None for item in evidence] == [True, True, True, True, False]


def test_items_that_are_not_objects_or_have_absurd_numbers_do_not_fail():
    evidence = [
        "not an item",
        {"box": [0, 0, float("inf"), 5]},
        {"text": 7, "value": "big", "point": [1e15, -1e15]},
        {"text": "far", "kind": "measured", "polygon": [[1e12, 0], [0, 1e12], [-1e12, -1e12]]},
    ]

    image = render_evidence(png(), evidence, focus=2)

    assert image.size == (400, 300)
    assert geometry(evidence[0]) is None and geometry(evidence[1]) is None
    assert [row["Item"] for row in legend_rows(evidence)] == ["", "", "7", "far"]


def test_a_transparent_drawing_is_put_on_white():
    image = render_evidence(png(mode="RGBA", colour=(0, 0, 0, 0)), [])

    assert image.mode == "RGB" and image.getpixel((5, 5)) == (255, 255, 255)


def test_focus_crops_to_the_item_with_a_margin():
    evidence = [{"text": "MH", "box": [100, 100, 140, 120]}, {"text": "Total", "kind": "total", "value": 1}]

    crop = render_evidence(png(size=(2000, 1500)), evidence, focus=0)

    assert crop.width < 2000 and crop.height < 1500
    assert crop.width >= 40 + 2 * 100                  # at least the margin on both sides
    assert render_evidence(png(), evidence, focus=1).size == (400, 300)   # nothing to zoom to


def test_legend_rows_list_every_item_in_order():
    rows = legend_rows([
        {"text": "5.06", "detail": "no name", "kind": "measured", "value": 2.0, "unit": "m²", "polygon": []},
        {"text": "MH", "box": [1, 2, 3, 4]},
        {"text": "Total", "kind": "total", "value": 83.51, "unit": "m²"},
    ])

    assert rows[0] == {"Item": "5.06", "Detail": "no name", "Value": "2.00 m²",
                       "How": "area measured by code from the room outline"}
    assert rows[1] == {"Item": "MH", "Detail": "", "Value": "", "How": "counted on the drawing"}
    assert rows[2]["Value"] == "83.51 m²" and rows[2]["How"] == "total"


def test_format_value_and_the_colour_key():
    assert format_value({"value": 3}) == "3"
    assert format_value({"value": 2.0, "unit": "m²"}) == "2.00 m²"
    assert format_value({}) == ""
    key = colour_key([
        {"text": "a", "kind": "printed", "point": [1, 1]}, {"text": "b", "kind": "printed", "point": [2, 2]},
        {"text": "c", "kind": "new", "point": [3, 3]}, {"text": "Total", "kind": "total"},
    ])
    assert key == [("blue", "area printed on the drawing"), ("gray", "new")]


def test_marks_of_the_llm_have_their_own_words_and_colour():
    assert describe_method("llm-located").startswith("counted by the LLM")
    item = {"text": "door 1", "kind": "llm", "point": [5, 5]}
    assert colour_key([item]) == [("green", "marked by the LLM as counted; code did not check the position")]
    image = render_evidence(png(), [item])
    assert image.size == (400, 300) and image.getextrema() != ((255, 255), (255, 255), (255, 255))


def test_method_words():
    assert describe_method("labels") == "counted from the labels printed on the drawing"
    assert "printed" in describe_method("printed+measured") and "measured" in describe_method("printed+measured")
    assert "LLM" in describe_method(None)
    assert describe_method("something-new") == "something-new"


def test_the_tag_of_a_narrow_room_is_made_small_enough_to_stay_in_the_room():
    narrow = [[100, 100], [180, 100], [180, 400], [100, 400]]
    wide = [[100, 100], [900, 100], [900, 400], [100, 400]]
    item = {"text": "5.14", "kind": "printed", "value": 13.21, "unit": "m²"}

    small = render_evidence(png(size=(2000, 1500)), [{**item, "polygon": narrow}])
    large = render_evidence(png(size=(2000, 1500)), [{**item, "polygon": wide}])

    assert all(small.getpixel((200, y)) == (255, 255, 255) for y in range(90, 160))   # right of the room
    def tag_pixels(image):
        return sum(image.getpixel((x, y)) == BLUE for x in range(104, 400) for y in range(104, 140))

    assert 0 < tag_pixels(small) < tag_pixels(large)                                  # there, but smaller


def test_the_box_of_a_small_label_on_a_large_drawing_does_not_cover_the_label():
    image = render_evidence(png(size=(2480, 1753)), [{"text": "Stanza", "box": [1044, 316, 1096, 328]}])

    inside = [image.getpixel((x, y)) for x in range(1045, 1096) for y in range(317, 328)]
    assert set(inside) == {(255, 255, 255)}
    assert image.getpixel((1044, 322)) == RED


def test_a_text_with_line_breaks_and_a_number_too_large_for_a_float_do_not_fail():
    evidence = [
        {"text": "two\nlines", "box": [100, 100, 200, 180]},
        {"text": "two\nlines " * 30, "kind": "measured", "polygon": [[10, 10], [90, 10], [90, 60]]},
        {"text": "huge", "value": 10**400, "point": [50, 50]},
        {"text": "far", "point": [10**400, 5]},
    ]

    image = render_evidence(png(), evidence)

    assert image.getpixel((100, 140)) == RED
    assert geometry(evidence[3]) is None
    assert [row["Value"] for row in legend_rows(evidence)] == ["", "", "", ""]
    assert len(colour_key(evidence)) == 2
