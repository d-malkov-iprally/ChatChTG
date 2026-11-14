"""Open an image file and bring it to the form that the detectors and the display need."""

from __future__ import annotations

from typing import Any

from PIL import Image, ImageOps


class UnreadableImageError(ValueError):
    """The file is not an image that can be opened."""


def to_rgb(image: Image.Image) -> Image.Image:
    """Return ``image`` as RGB. Transparent pixels become white, as a viewer shows a drawing.

    A plain ``convert("RGB")`` drops the alpha channel. A transparent pixel is usually stored
    as black, so the background of a drawing would become as dark as its lines.
    """
    if image.mode.startswith("I;16"):
        # convert() does not scale 16-bit grey values to 8 bits: all but the darkest turn white.
        import numpy as np

        return Image.fromarray((np.asarray(image) >> 8).astype(np.uint8)).convert("RGB")
    if image.mode in ("RGBA", "LA", "PA") or "transparency" in image.info:
        rgba = image.convert("RGBA")
        flat = Image.new("RGB", rgba.size, "white")
        flat.paste(rgba, mask=rgba.getchannel("A"))
        return flat
    return image.convert("RGB")


def open_image(source: Any) -> Image.Image:
    """Open a path or a file object as an upright RGB image.

    The rotation that a camera stored in the file's EXIF data is applied.
    Raise UnreadableImageError when the content is not an image, is cut off or is too large.
    """
    try:
        with Image.open(source) as image:
            return to_rgb(ImageOps.exif_transpose(image))
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as error:
        raise UnreadableImageError(f"This file cannot be opened as an image: {error}") from error
