"""Evidence: where an answer comes from on the drawing, and how to show it.

This module is the home of the evidence contract. ``AnswerResponse.evidence`` is a list of
items. An item is a JSON object. Coordinates are pixels of the stored image (the bytes of
the document, decoded without EXIF rotation), origin top left, x to the right, y down;
by convention rounded to whole pixels.

==========  ===================================  ==========================================
Key         Type                                 Meaning
==========  ===================================  ==========================================
``text``    string, required                     Short name, drawn as the tag: a label that
                                                 was read, a room number, joined numbers of
                                                 a shared region (``5.15+5.00``), ``Total``.
``box``     ``[x0, y0, x1, y1]``                 A rectangle.
``polygon`` ``[[x, y], ...]``, 3 or more points  A closed outline; the last point is not
                                                 repeated.
``point``   ``[x, y]``                           A place without an extent.
``kind``    string, optional                     Legend group, it selects the colour. Known:
                                                 ``printed``, ``measured``, ``shared``,
                                                 ``total``. Absent: counted on the drawing.
                                                 An unknown kind gets the fallback colour
                                                 and is named by its own string.
``value``   number, optional                     The number the item contributes (an area).
``unit``    string, optional                     Unit of ``value`` (``m²``).
``detail``  string, optional                     Longer words, for the list only.
==========  ===================================  ==========================================

Rules:

- An item has one of ``box``, ``polygon``, ``point``, or none. With more than one, the
  renderer draws one: ``polygon``, else ``box``, else ``point``.
- An item without geometry is a fact without a place (the total). It is listed, never drawn.
- Readers ignore unknown keys. The renderer skips an item whose geometry is malformed and
  never raises for it; coordinates outside the image are clipped by the drawing.
- ``AnswerResponse.method`` is a short code; its words are in ``METHOD_WORDS``. A producer
  that adds a method adds its words there.
- One response carries the evidence of one producer.

The module needs Pillow only, so the app, the eval tools and the tests can all use it.
"""

from __future__ import annotations

import io
import math
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from chatchtg.detection.images import to_rgb

METHOD_WORDS = {
    "labels": "counted from the labels printed on the drawing",
    "shapes": "counted from the shapes on the drawing",
    "labels+shapes": "counted from the printed labels and confirmed by the shapes",
    "usual": "not counted on the drawing: this is the usual number for such a plan",
    "printed": "added up in code from the areas printed on the drawing",
    "measured": "added up in code from room outlines that code measured on the drawing",
    "printed+measured": (
        "added up in code: printed areas where the drawing has them, measured outlines for the other rooms"
    ),
}
NO_METHOD_WORDS = "answered by the LLM; code did not locate the answer on the drawing"

# kind -> (words, Streamlit colour name, RGB). None is an item without a kind.
KINDS: dict[str | None, tuple[str, str, tuple[int, int, int]]] = {
    None: ("counted on the drawing", "red", (230, 25, 75)),
    "printed": ("area printed on the drawing", "blue", (31, 119, 180)),
    "measured": ("area measured by code from the room outline", "orange", (245, 130, 49)),
    "shared": ("rooms measured as one region: no wall was found between them", "violet", (145, 30, 180)),
}
FALLBACK_COLOUR = ("green", (44, 160, 44))  # an unknown kind; its words are the kind itself
FILL_ALPHA = 40  # of 255: the printed labels stay readable under the fill

# The line and the tag grow with the image, because the app shows large drawings scaled down.
LINE_WIDTH = 1.5  # pixels for each 1000 pixels of the longer side
FONT_SIZE = 9
MIN_FONT_SIZE = 11
# The tag of a room is made smaller until it takes at most this part of the room's width, so
# that the tags of a dense plan (1:200, 40 rooms) do not cover each other and the labels.
TAG_PART_OF_WIDTH = 0.7
SMALLEST_FONT_SIZE = 5  # the limit of that, again for each 1000 pixels; MIN_FONT_SIZE in pixels
POINT_RADIUS = 8
FOCUS_MARGIN = 100  # pixels around an item in the zoomed view, at least

_FAR = 100_000  # coordinates are limited to this before drawing: Pillow is slow on huge shapes


def describe_method(method: str | None) -> str:
    """The words for a response's method. An unknown method is returned as it is."""
    if method is None:
        return NO_METHOD_WORDS
    return METHOD_WORDS.get(method, method)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _numbers(value: Any, length: int) -> bool:
    return isinstance(value, (list, tuple)) and len(value) == length and all(_is_number(v) for v in value)


def geometry(item: Any) -> tuple[str, list] | None:
    """The one geometry of an item to draw: ``("polygon", points)``, ``("box", [x0, y0, x1, y1])``
    or ``("point", [x, y])``. None when the item has no valid geometry.

    Anything malformed counts as absent. The corners of a box are put in order.
    """
    if not isinstance(item, dict):
        return None
    polygon = item.get("polygon")
    if isinstance(polygon, (list, tuple)) and len(polygon) >= 3 and all(_numbers(p, 2) for p in polygon):
        return "polygon", [[p[0], p[1]] for p in polygon]
    box = item.get("box")
    if _numbers(box, 4):
        return "box", [min(box[0], box[2]), min(box[1], box[3]), max(box[0], box[2]), max(box[1], box[3])]
    point = item.get("point")
    if _numbers(point, 2):
        return "point", [point[0], point[1]]
    return None


def _number_text(value: Any) -> str:
    if not _is_number(value):
        return ""
    return str(value) if isinstance(value, int) else f"{value:.2f}"


def format_value(item: dict[str, Any]) -> str:
    """The value of an item with its unit: ``3``, ``2.00 m²``. Empty without a value."""
    number = _number_text(item.get("value"))
    unit = item.get("unit")
    return f"{number} {unit}" if number and isinstance(unit, str) and unit else number


def _kind(item: Any) -> str | None:
    kind = item.get("kind") if isinstance(item, dict) else None
    return kind if isinstance(kind, str) else None


def _kind_words(kind: str | None) -> str:
    if kind in KINDS:
        return KINDS[kind][0]
    return str(kind)


def _kind_colour(kind: str | None) -> tuple[str, tuple[int, int, int]]:
    if kind in KINDS:
        return KINDS[kind][1], KINDS[kind][2]
    return FALLBACK_COLOUR


def _text(item: Any) -> str:
    text = item.get("text") if isinstance(item, dict) else None
    return "" if text is None else str(text)


def legend_rows(evidence: list[dict[str, Any]]) -> list[dict[str, str]]:
    """One table row for each item, in order: what it is, its value, and how code got it."""
    rows = []
    for item in evidence:
        detail = item.get("detail") if isinstance(item, dict) else None
        rows.append(
            {
                "Item": _text(item),
                "Detail": str(detail) if detail else "",
                "Value": format_value(item) if isinstance(item, dict) else "",
                "How": _kind_words(_kind(item)),
            }
        )
    return rows


def colour_key(evidence: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """``(Streamlit colour name, words)`` for each kind that is drawn, in order of first appearance."""
    key: list[tuple[str, str]] = []
    for item in evidence:
        if geometry(item) is None:
            continue
        entry = (_kind_colour(_kind(item))[0], _kind_words(_kind(item)))
        if entry not in key:
            key.append(entry)
    return key


def _limit(value: float) -> int:
    return round(max(-_FAR, min(_FAR, value)))


def _bounds(shape: str, data: list, radius: int) -> tuple[int, int, int, int]:
    if shape == "polygon":
        xs, ys = [_limit(p[0]) for p in data], [_limit(p[1]) for p in data]
        return min(xs), min(ys), max(xs), max(ys)
    if shape == "box":
        return _limit(data[0]), _limit(data[1]), _limit(data[2]), _limit(data[3])
    x, y = _limit(data[0]), _limit(data[1])
    return x - radius, y - radius, x + radius, y + radius


def _tag_text(item: dict[str, Any]) -> str:
    return " ".join(part for part in (_text(item), _number_text(item.get("value"))) if part)


def _fitted_font(draw: ImageDraw.ImageDraw, tag: str, size: int, smallest: int, room: float):
    """The default font in ``size``, or smaller so that ``tag`` is at most ``room`` pixels wide."""
    font = ImageFont.load_default(size=size)
    width = draw.textlength(tag, font=font)
    if width <= room or size <= smallest:
        return font
    return ImageFont.load_default(size=max(smallest, int(size * room / width)))


def _draw_tag(draw: ImageDraw.ImageDraw, at: tuple[float, float], above: bool, tag: str, font, colour) -> None:
    """White text on a rectangle of the item's colour, kept inside the image."""
    if not tag:
        return
    left, top, right, bottom = draw.textbbox(at, tag, font=font, anchor="lb" if above else "lt")
    if above and top < 0:
        # No room above the item: put the tag inside it.
        left, top, right, bottom = draw.textbbox(at, tag, font=font, anchor="lt")
    width, height = draw.im.size
    shift_x = min(0, width - right) if left >= 0 else -left
    shift_y = min(0, height - bottom) if top >= 0 else -top
    left, top, right, bottom = left + shift_x, top + shift_y, right + shift_x, bottom + shift_y
    draw.rectangle((left, top, right, bottom), fill=colour)
    draw.text((left, top), tag, fill="white", font=font, anchor="lt")


def render_evidence(content: bytes, evidence: list[dict[str, Any]], focus: int | None = None) -> Image.Image:
    """Return the drawing as an RGB image with the evidence items drawn on it.

    ``content`` is the stored file. It is decoded without EXIF rotation, because the
    coordinates of the items refer to the stored pixels, as the pipelines read them.
    With ``focus``, the index of an item that has a geometry, the result is cut to that
    item and a margin around it. Any other ``focus`` gives the whole drawing.
    An item that cannot be drawn is skipped.
    """
    with Image.open(io.BytesIO(content)) as opened:
        image = to_rgb(opened)
    scale = max(image.size) / 1000
    width = max(2, round(LINE_WIDTH * scale))
    radius = max(POINT_RADIUS, round(POINT_RADIUS * scale))
    font_size = max(MIN_FONT_SIZE, round(FONT_SIZE * scale))
    smallest = min(font_size, max(MIN_FONT_SIZE, round(SMALLEST_FONT_SIZE * scale)))
    font = ImageFont.load_default(size=font_size)

    shapes = []  # (item, shape, data, colour) of the items that can be drawn
    for item in evidence:
        found = geometry(item)
        if found is not None:
            shapes.append((item, *found, _kind_colour(_kind(item))[1]))

    # The fills first, on a transparent layer, so that the drawing shows through them.
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    fills = ImageDraw.Draw(overlay)
    for item, shape, data, colour in shapes:
        if _kind(item) is None:
            continue  # a counted label: the outline only, so the label stays as printed
        if shape == "polygon":
            fills.polygon([(_limit(x), _limit(y)) for x, y in data], fill=(*colour, FILL_ALPHA))
        elif shape == "box":
            fills.rectangle(_bounds(shape, data, radius), fill=(*colour, FILL_ALPHA))
    annotated = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")

    draw = ImageDraw.Draw(annotated)
    grow = width - 1
    for item, shape, data, colour in shapes:
        left, top, right, bottom = _bounds(shape, data, radius)
        if shape == "polygon":
            points = [(_limit(x), _limit(y)) for x, y in data]
            draw.line([*points, points[0], points[1]], fill=colour, width=width, joint="curve")
        elif shape == "box":
            # The line goes outwards from the box, so that it does not cover a small label.
            draw.rectangle((left - grow, top - grow, right + grow, bottom + grow), outline=colour, width=width)
        else:
            draw.ellipse((left, top, right, bottom), outline=colour, width=width)
    # The tags last, so that no outline crosses a tag.
    for item, shape, data, colour in shapes:
        left, top, right, bottom = _bounds(shape, data, radius)
        tag = _tag_text(item)
        if shape == "polygon":
            # In the corner of the room, not on its centre: the printed label is there.
            x, y = min(data, key=lambda p: (_limit(p[0]) - left) ** 2 + (_limit(p[1]) - top) ** 2)
            fitted = _fitted_font(draw, tag, font_size, smallest, TAG_PART_OF_WIDTH * (right - left))
            _draw_tag(draw, (_limit(x) + width, _limit(y) + width), False, tag, fitted, colour)
        else:
            lift = grow if shape == "box" else 0
            _draw_tag(draw, (left - lift, top - lift), True, tag, font, colour)

    if focus is None or isinstance(focus, bool) or not isinstance(focus, int) or not 0 <= focus < len(evidence):
        return annotated
    found = geometry(evidence[focus])
    if found is None:
        return annotated
    left, top, right, bottom = _bounds(*found, radius)
    margin = max(FOCUS_MARGIN, right - left, bottom - top)
    crop = (
        max(0, left - margin), max(0, top - margin),
        min(annotated.width, right + margin), min(annotated.height, bottom + margin),
    )
    if crop[0] >= crop[2] or crop[1] >= crop[3]:
        return annotated  # the item is outside the drawing
    return annotated.crop(crop)
