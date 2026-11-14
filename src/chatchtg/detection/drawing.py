"""Draw detections on an image."""

from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

from chatchtg.detection.detector import Detection

# Colours that stay visible on a white drawing with black lines.
PALETTE = ["#e6194b", "#1f77b4", "#2ca02c", "#f58231", "#911eb4", "#008080", "#9a6324", "#d62790"]


def draw_detections(image: Image.Image, detections: list[Detection]) -> Image.Image:
    """Return a copy of ``image`` with one box and one "label score" tag per detection.

    Each label gets its own colour, in the order in which the labels first occur.
    """
    annotated = image.convert("RGB")
    draw = ImageDraw.Draw(annotated)
    # Scale the line and the text with the image: the app shows large drawings scaled down.
    scale = max(annotated.size) / 1000
    width = max(2, round(2 * scale))
    font = ImageFont.load_default(size=max(12, round(14 * scale)))
    colours: dict[str, str] = {}
    for detection in detections:
        colour = colours.setdefault(detection.label, PALETTE[len(colours) % len(PALETTE)])
        left, top, right, bottom = detection.box
        draw.rectangle((left, top, right, bottom), outline=colour, width=width)
        tag = f"{detection.label} {detection.score:.2f}"
        tag_box = draw.textbbox((left, top), tag, font=font, anchor="lb")
        if tag_box[1] < 0:
            # No room above the box: put the tag inside it.
            tag_box = draw.textbbox((left, top), tag, font=font, anchor="lt")
        draw.rectangle(tag_box, fill=colour)
        draw.text((tag_box[0], tag_box[1]), tag, fill="white", font=font, anchor="lt")
    return annotated
