"""Read the text of a floor plan with Apple Vision (macOS only), with a file cache.

Boxes are in the pixels of the original image: (x0, y0, x1, y1), y down.
A Finnish plan is read whole in four rotations, mirrored too, and in tiles. A Swiss sheet is
large with small text, so it is read in tiles at two scales.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import sys
from pathlib import Path
from typing import Any

from PIL import Image

FINNISH_LANGS = ("fi-FI", "en-US")
SWISS_LANGS = ("de-DE", "fr-FR", "it-IT", "en-US")
PROFILES = ("finnish", "swiss")
_VERSION = 1
"""Increase when the passes change, so cached results are not reused."""

Ocr = dict[str, Any]
Token = dict[str, Any]


class OcrUnavailableError(RuntimeError):
    """Apple Vision cannot be loaded on this machine."""


def available() -> bool:
    if sys.platform != "darwin":
        return False
    try:
        import Vision  # noqa: F401
    except ImportError:
        return False
    return True


def read_plan(content: bytes, profile: str, cache_dir: Path | None = None) -> Ocr:
    """Return the text tokens of one drawing. ``profile`` is ``finnish`` or ``swiss``."""
    if profile not in PROFILES:
        raise ValueError(f"Unknown OCR profile {profile!r}")
    cache_file = None
    if cache_dir is not None:
        digest = hashlib.sha256(content).hexdigest()
        cache_file = cache_dir / f"{digest}-{profile}-v{_VERSION}.json"
        if cache_file.is_file():
            return json.loads(cache_file.read_text(encoding="utf-8"))
    image = Image.open(io.BytesIO(content)).convert("RGB")
    result = _read_finnish(image) if profile == "finnish" else _read_swiss(image)
    result |= {"W": image.width, "H": image.height, "profile": profile}
    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_file.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        temporary.replace(cache_file)
    return result


# About 5% of the Finnish plans are stored mirrored. Their labels read only in the flipped image.
_ROOM_WORD = re.compile(
    r"^(MH\d*|OH|ET|KH|KPH|PH|PSH|VH|WC|KK|SAUNA|PARVEKE|ETEINEN|MAKUUH|KEITTIÖ|TK|KT|RT|KHH|VAR|TH|AH|PARV)$"
)


def _room_words(tokens: list[Token]) -> int:
    from chatchtg.rooms.lexicon import words

    return sum(
        1 for t in tokens if t["conf"] >= 0.5 for w in words(t["text"]) if _ROOM_WORD.match(w)
    )


def _read_finnish(image: Image.Image) -> Ocr:
    rotations = (0, 90, 180, 270)
    whole = _tag(_whole(image, rotations, FINNISH_LANGS, mirror=False), "whole")
    flipped = _tag(_whole(image, rotations, FINNISH_LANGS, mirror=True), "mirror")
    if _room_words(flipped) > _room_words(whole):
        return {"tokens": flipped, "mirrored": True}
    tiles = _tiled(image, rotations, tile=700, overlap=120, scale=2.0, langs=FINNISH_LANGS)
    return {"tokens": whole + _tag(tiles, "tiles"), "mirrored": False}


def _read_swiss(image: Image.Image) -> Ocr:
    one = _tiled(image, (0,), tile=900, overlap=150, scale=1.0, langs=SWISS_LANGS)
    two = _tiled(image, (0,), tile=900, overlap=150, scale=2.0, langs=SWISS_LANGS)
    return {"tokens": _tag(one, "tiles1") + _tag(two, "tiles2"), "mirrored": False}


def _tag(tokens: list[Token], source: str) -> list[Token]:
    return [{**t, "src": source} for t in tokens]


def _whole(
    image: Image.Image, rotations: tuple[int, ...], langs: tuple[str, ...], mirror: bool
) -> list[Token]:
    """Read the whole image, enlarged to about 2000 pixels on its long side."""
    width, height = image.size
    scale = max(1.0, min(3.0, 2000 / max(width, height)))
    big = image
    if scale != 1:
        big = image.resize((round(width * scale), round(height * scale)), Image.LANCZOS)
    if mirror:
        big = big.transpose(Image.FLIP_LEFT_RIGHT)
    tokens = []
    for rotation in rotations:
        turned = big.rotate(rotation, expand=True) if rotation else big
        for token in _vision(turned, langs):
            box = _unrotate(token["box"], rotation, width, height, scale)
            if mirror:
                box = [width - box[2], box[1], width - box[0], box[3]]
            tokens.append({**token, "box": box, "rot": rotation})
    return tokens


def _tiled(
    image: Image.Image,
    rotations: tuple[int, ...],
    tile: int,
    overlap: int,
    scale: float,
    langs: tuple[str, ...],
) -> list[Token]:
    """Read the image in overlapping tiles, each enlarged by ``scale``."""
    width, height = image.size
    tokens = []
    for top in range(0, max(1, height - overlap), tile - overlap):
        for left in range(0, max(1, width - overlap), tile - overlap):
            crop = image.crop((left, top, min(width, left + tile), min(height, top + tile)))
            if crop.convert("L").getextrema()[0] > 200:  # a blank tile
                continue
            crop_width, crop_height = crop.size
            big = crop
            if scale != 1:
                size = (round(crop_width * scale), round(crop_height * scale))
                big = crop.resize(size, Image.LANCZOS)
            for rotation in rotations:
                turned = big.rotate(rotation, expand=True) if rotation else big
                for token in _vision(turned, langs):
                    box = _unrotate(token["box"], rotation, crop_width, crop_height, scale)
                    box = [box[0] + left, box[1] + top, box[2] + left, box[3] + top]
                    tokens.append({**token, "box": box, "rot": rotation})
    return _drop_duplicates(tokens)


def _drop_duplicates(tokens: list[Token]) -> list[Token]:
    """Drop a token that lies mostly inside a longer token of the same rotation."""
    kept: list[Token] = []
    for token in sorted(tokens, key=lambda t: -(len(t["text"]) + t["conf"])):
        b = token["box"]
        area = (b[2] - b[0]) * (b[3] - b[1])
        duplicate = False
        for other in kept:
            a = other["box"]
            overlap_x = max(0, min(a[2], b[2]) - max(a[0], b[0]))
            overlap_y = max(0, min(a[3], b[3]) - max(a[1], b[1]))
            if other["rot"] == token["rot"] and area > 0 and overlap_x * overlap_y / area > 0.6:
                duplicate = True
                break
        if not duplicate:
            kept.append(token)
    return kept


def _unrotate(box: list[float], rotation: int, width: int, height: int, scale: float) -> list[float]:
    """Map a box of the enlarged, rotated image back to the original image."""
    x0, y0, x1, y1 = (v / scale for v in box)
    points = []
    for x, y in ((x0, y0), (x1, y0), (x0, y1), (x1, y1)):
        if rotation == 90:
            x, y = width - y, x
        elif rotation == 180:
            x, y = width - x, height - y
        elif rotation == 270:
            x, y = y, height - x
        points.append((x, y))
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def _vision(image: Image.Image, langs: tuple[str, ...]) -> list[Token]:
    try:
        import Vision
        from Foundation import NSData
    except ImportError as error:
        raise OcrUnavailableError(
            "Reading labels needs Apple Vision (macOS with pyobjc-framework-Vision)."
        ) from error
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    data = NSData.dataWithBytes_length_(buffer.getvalue(), len(buffer.getvalue()))
    handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(data, None)
    request = Vision.VNRecognizeTextRequest.alloc().init()
    request.setRecognitionLevel_(0)  # accurate
    request.setUsesLanguageCorrection_(False)
    request.setRecognitionLanguages_(list(langs))
    request.setMinimumTextHeight_(0.0)
    handler.performRequests_error_([request], None)
    width, height = image.size
    tokens = []
    for observation in request.results() or []:
        candidate = observation.topCandidates_(1)[0]
        bounds = observation.boundingBox()
        x, y = bounds.origin.x, bounds.origin.y
        w, h = bounds.size.width, bounds.size.height
        tokens.append(
            {
                "text": str(candidate.string()),
                "conf": float(candidate.confidence()),
                "box": [x * width, (1 - y - h) * height, (x + w) * width, (1 - y) * height],
            }
        )
    return tokens
