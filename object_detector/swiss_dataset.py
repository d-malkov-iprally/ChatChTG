"""The Swiss Dwellings dataset of the detector: labels from the DXF twins, the split, the tiles.

Run on a machine that has the data (the DXF files are not uploaded to Modal):

    uv run python -m object_detector.swiss_dataset labels --out var/swiss-detector/labels.json
    uv run python -m object_detector.swiss_dataset draw --labels var/swiss-detector/labels.json \
        --out var/swiss-detector/drawn site1000_bldg2082_floor9385

``labels.json`` holds, for each plan: its split, its scale, the size of the PNG, the fitted
transform and the boxes in PNG pixels. ``modal_swiss.py`` uploads it with the PNG files and
cuts the tiles on the volume.

The DXF files are training labels and a reference for scores. The plans of the test split
are never trained on and never used to choose a threshold (ADR 0003).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterator, Mapping
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from object_detector.cubicasa_svg import Box
from object_detector.swiss_dxf import ASKED_CLASSES, SWISS_CLASSES

PNG_DIR = Path("data/floorplans/data/swiss_dwellings")
DXF_DIR = Path("data/full/data/swiss_dwellings")
QUESTIONS = Path("data/floorplans/questions.jsonl")
BENCHMARK_FOLDER = "data/swiss_dwellings/"

# The start of the English question of each template -> the class that it counts.
TEMPLATES = {
    "How many doors (interior doors and entrance doors together)": "door",
    "How many windows are drawn": "window",
    "How many toilets (WC) are drawn": "toilet",
    "How many washbasins are drawn": "washbasin",
    "How many bathtubs are drawn": "bathtub",
    "How many showers are drawn": "shower",
}

SPLIT_SHARES = (("train", 0.70), ("valid", 0.15), ("test", 0.15))
STEM = re.compile(r"^site(\d+)_bldg(\d+)_floor(\d+)$")

TILE = 640
"""Side of a training tile, in PNG pixels. The tiles are not scaled: a 1:200 toilet is about
12 x 18 pixels, too small to be seen in a whole sheet that is scaled to 1024."""
STRIDE = 448
"""The tiles overlap by TILE - STRIDE = 192 pixels, so an object up to 192 pixels wide is
whole in at least one tile."""
MIN_VISIBLE = 0.5
"""A box that is cut by the edge of a tile is kept if this part of it is in the tile."""


# --- split ---------------------------------------------------------------------------------


def site_of(stem: str) -> str:
    match = STEM.match(stem)
    if not match:
        raise ValueError(f"Not a Swiss Dwellings plan name: {stem}")
    return match.group(1)


def split_of(stem: str) -> str:
    """train, valid or test, from a hash of the site: all floors of a site are in one split."""
    digest = hashlib.sha256(f"swiss-site-{site_of(stem)}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    total = 0.0
    for split, share in SPLIT_SHARES:
        total += share
        if value < total:
            return split
    return SPLIT_SHARES[-1][0]


# --- drawing area --------------------------------------------------------------------------


def drawing_area(grey: np.ndarray) -> tuple[int, int, int, int]:
    """(left, top, right, bottom) of the drawing in a sheet, from the PNG only.

    The sheet has a frame and a title block with a legend under the drawing. The legend
    shows one of each symbol, and those must not be counted. The frame and the top of the
    title block are the rows and columns that are dark over most of the sheet.
    """
    height, width = grey.shape
    dark = grey < 100
    rows = np.flatnonzero(dark.mean(axis=1) > 0.7)
    cols = np.flatnonzero(dark.mean(axis=0) > 0.7)

    def end_of_first_run(values: np.ndarray, limit: float) -> int:
        """The index after the first run of adjacent values (the frame line), or 0."""
        values = values[values < limit]
        if not values.size:
            return 0
        end = values[0]
        while end + 1 in values:
            end += 1
        return int(end) + 1

    top = end_of_first_run(rows, height / 4)
    left = end_of_first_run(cols, width / 4)
    right_cols = cols[cols > 3 * width / 4]
    right = width - end_of_first_run(width - 1 - right_cols[::-1], width / 4) if right_cols.size else width
    # The top line of the title block is the lowest long row above the bottom edge of the
    # frame. A long wall in the drawing is above it.
    inner = rows[(rows > height / 2) & (rows < height - height / 20)]
    bottom = int(inner.max()) if inner.size else height
    while bottom - 1 in inner:
        bottom -= 1
    return left, top, right, bottom


def tiles(area: tuple[int, int, int, int], tile: int = TILE, stride: int = STRIDE) -> list[tuple[int, int]]:
    """The top-left corners of the tiles that cover ``area``. The last tile of a row or
    column is moved back so that it ends on the edge of the area."""

    def starts(low: int, high: int) -> list[int]:
        if high - low <= tile:
            return [low]
        values = list(range(low, high - tile, stride))
        values.append(high - tile)
        return values

    left, top, right, bottom = area
    return [(x, y) for y in starts(top, bottom) for x in starts(left, right)]


def boxes_in_tile(boxes: list[Box], x: int, y: int, tile: int = TILE) -> list[Box]:
    """The boxes in tile coordinates, cut to the tile. A box is kept if at least MIN_VISIBLE
    of its area is in the tile."""
    out = []
    for box in boxes:
        left, top = max(box.left, x), max(box.top, y)
        right, bottom = min(box.right, x + tile), min(box.bottom, y + tile)
        if right <= left or bottom <= top:
            continue
        area = max((box.right - box.left) * (box.bottom - box.top), 1e-9)
        if (right - left) * (bottom - top) >= MIN_VISIBLE * area:
            out.append(Box(box.label, left - x, top - y, right - x, bottom - y))
    return out


def core(x: int, y: int, area: tuple[int, int, int, int], tile: int = TILE, stride: int = STRIDE):
    """The part of a tile whose detections are kept when the tiles of a sheet are joined:
    the tile without half the overlap on each side that has a neighbour."""
    half = (tile - stride) // 2
    left, top, right, bottom = area
    return (
        x if x <= left else x + half,
        y if y <= top else y + half,
        x + tile if x + tile >= right else x + tile - half,
        y + tile if y + tile >= bottom else y + tile - half,
    )


def join_tiles(found: list[tuple[tuple[int, int], list[dict]]], area, tile: int = TILE,
               stride: int = STRIDE) -> list[dict]:
    """Join the detections of the tiles of one sheet.

    ``found`` holds, for each tile, its corner and its detections in tile pixels
    (``label``, ``score``, ``box`` as x0, y0, x1, y1). A detection is kept by the tile whose
    core holds the centre of its box. The cores of the tiles overlap where a last tile was
    moved back, so overlapping boxes of one class are then reduced to the best one.
    """
    kept = []
    for (x, y), detections in found:
        c = core(x, y, area, tile, stride)
        for d in detections:
            box = [d["box"][0] + x, d["box"][1] + y, d["box"][2] + x, d["box"][3] + y]
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            if c[0] <= cx < c[2] and c[1] <= cy < c[3]:
                kept.append({**d, "box": box})
    return non_max_suppression(kept)


def _iou(a, b) -> float:
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    if width <= 0 or height <= 0:
        return 0.0
    inter = width * height
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union


def non_max_suppression(detections: list[dict], iou: float = 0.5) -> list[dict]:
    out: list[dict] = []
    for d in sorted(detections, key=lambda d: -d["score"]):
        if all(o["label"] != d["label"] or _iou(o["box"], d["box"]) < iou for o in out):
            out.append(d)
    return out


# --- labels --------------------------------------------------------------------------------


def benchmark_answers(questions: Path = QUESTIONS) -> dict[str, dict[str, int]]:
    """Plan stem -> class -> answer, for the English object-count questions of the Swiss plans.

    A plan without a bathtub or shower question has none of them: the question is asked only
    when the answer is not 0 (checked against the DXF labels on all plans).
    """
    answers: dict[str, dict[str, int]] = {}
    with questions.open(encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            if not row["path"].startswith(BENCHMARK_FOLDER):
                continue
            for start, label in TEMPLATES.items():
                if row["question"].startswith(start):
                    stem = Path(row["path"]).stem
                    answers.setdefault(stem, {})[label] = int(row["answer"])
    return answers


def _label_plan(args: tuple[str, str, str]) -> tuple[str, dict]:
    from PIL import Image

    from object_detector.swiss_dxf import boxes_in_pixels, fit_transform, read_drawing, read_symbols

    stem, png_dir, dxf_dir = args
    with Image.open(Path(png_dir) / f"{stem}.png") as picture:
        image = np.asarray(picture.convert("RGB"))
    drawing = read_drawing(Path(dxf_dir) / f"{stem}.dxf")
    transform, fit = fit_transform(image, drawing)
    height, width = image.shape[:2]
    boxes = boxes_in_pixels(read_symbols(drawing), transform, (width, height))
    area = drawing_area(image.mean(axis=2))
    outside = sum(
        1 for b in boxes
        if (b.left + b.right) / 2 < area[0] or (b.left + b.right) / 2 > area[2]
        or (b.top + b.bottom) / 2 < area[1] or (b.top + b.bottom) / 2 > area[3]
    )
    return stem, {
        "split": split_of(stem),
        "scale": drawing.scale,
        "width": width,
        "height": height,
        "area": list(area),
        "transform": [transform.scale, transform.dx, transform.dy],
        "fit": round(fit, 4),
        "boxes_outside_area": outside,
        "boxes": [[b.label, round(b.left, 1), round(b.top, 1), round(b.right, 1), round(b.bottom, 1)]
                  for b in boxes],
    }


def build_labels(png_dir: Path = PNG_DIR, dxf_dir: Path = DXF_DIR, workers: int = 8) -> dict[str, dict]:
    stems = sorted(p.stem for p in png_dir.glob("*.png"))
    jobs = [(stem, str(png_dir), str(dxf_dir)) for stem in stems]
    with ProcessPoolExecutor(workers) as pool:
        return dict(pool.map(_label_plan, jobs, chunksize=4))


def plan_boxes(plan: Mapping) -> list[Box]:
    return [Box(row[0], *row[1:]) for row in plan["boxes"]]


def counts_of(plan: Mapping) -> Counter[str]:
    return Counter(row[0] for row in plan["boxes"])


def check_labels(labels: Mapping[str, Mapping], answers: Mapping[str, Mapping[str, int]]) -> dict:
    """For each template: on how many plans the label count equals the benchmark answer.

    Only plans that have the question are scored; the plans without a bathtub or shower
    question are also checked to have no such label.
    """
    result = {}
    for label in TEMPLATES.values():
        asked = [s for s in labels if label in answers.get(s, {})]
        exact = [s for s in asked if counts_of(labels[s])[label] == answers[s][label]]
        unasked = [s for s in labels if label not in answers.get(s, {})]
        result[label] = {
            "plans": len(asked),
            "exact": len(exact),
            "exact_rate": len(exact) / len(asked) if asked else None,
            "wrong": sorted(set(asked) - set(exact))[:20],
            "unasked_plans": len(unasked),
            "unasked_with_label": sum(1 for s in unasked if counts_of(labels[s])[label]),
        }
    return result


# --- COCO tiles ----------------------------------------------------------------------------


def sheet_tiles(grey: np.ndarray, area) -> list[tuple[int, int]]:
    """The tiles of a sheet that have ink. A blank tile has nothing to learn or to find."""
    return [
        (x, y) for x, y in tiles(tuple(area))
        if (grey[y:y + TILE, x:x + TILE] < 200).mean() >= 0.002
    ]


def _write_plan_tiles(stem: str, plan: Mapping, png_dir: Path, split_dir: Path) -> list[tuple[str, list[Box]]]:
    from PIL import Image

    boxes = plan_boxes(plan)
    with Image.open(png_dir / f"{stem}.png") as picture:
        sheet = picture.convert("RGB")
    out = []
    for x, y in sheet_tiles(np.asarray(sheet.convert("L")), plan["area"]):
        name = f"{stem}_x{x}_y{y}.png"
        sheet.crop((x, y, x + TILE, y + TILE)).save(split_dir / name)
        out.append((name, boxes_in_tile(boxes, x, y)))
    return out


def coco_tiles(labels: Mapping[str, Mapping], png_dir: Path, out_dir: Path, workers: int = 16) -> dict:
    """Write the tiles of the plans to ``out_dir/<split>/`` with ``_annotations.coco.json``.

    Tiles with no ink are left out. Returns the number of tiles and boxes of each split.
    """
    from concurrent.futures import ThreadPoolExecutor

    stats: dict = {}
    by_split: dict[str, list[str]] = {}
    for stem, plan in sorted(labels.items()):
        by_split.setdefault(plan["split"], []).append(stem)
    categories = [{"id": i, "name": n, "supercategory": "symbol"} for i, n in enumerate(SWISS_CLASSES)]
    for split, stems in sorted(by_split.items()):
        split_dir = out_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        images, annotations = [], []
        with ThreadPoolExecutor(workers) as pool:
            written = list(pool.map(lambda s: _write_plan_tiles(s, labels[s], png_dir, split_dir), stems))
        for name, tile_boxes in (row for plan_rows in written for row in plan_rows):
            image_id = len(images)
            images.append({"id": image_id, "file_name": name, "width": TILE, "height": TILE})
            for box in tile_boxes:
                w, h = box.right - box.left, box.bottom - box.top
                annotations.append({
                    "id": len(annotations), "image_id": image_id,
                    "category_id": SWISS_CLASSES.index(box.label),
                    "bbox": [round(box.left, 1), round(box.top, 1), round(w, 1), round(h, 1)],
                    "area": round(w * h, 1), "iscrowd": 0,
                })
        coco = {"images": images, "annotations": annotations, "categories": categories}
        (split_dir / "_annotations.coco.json").write_text(json.dumps(coco))
        stats[split] = {
            "plans": len(stems),
            "tiles": len(images),
            "boxes": dict(Counter(SWISS_CLASSES[a["category_id"]] for a in annotations)),
        }
        print(f"{split}: {len(stems)} plans, {len(images)} tiles", flush=True)
    return stats


# --- scores --------------------------------------------------------------------------------


def score_plans(counts: Mapping[str, Mapping[str, int]], labels: Mapping[str, Mapping],
                answers: Mapping[str, Mapping[str, int]], stems: list[str]) -> dict:
    """For each asked class and each scale (all, 1:100, 1:200): on how many plans the count
    equals the benchmark answer, and the mean absolute error. Only plans with the question
    are scored, as in the benchmark."""
    result: dict = {}
    for label in ASKED_CLASSES:
        result[label] = {}
        for group in ("all", 100, 200):
            errors = [
                abs(counts.get(s, {}).get(label, 0) - answers[s][label])
                for s in stems
                if label in answers.get(s, {}) and (group == "all" or labels[s]["scale"] == group)
            ]
            result[label][str(group)] = {
                "plans": len(errors),
                "exact": sum(e == 0 for e in errors),
                "exact_rate": sum(e == 0 for e in errors) / len(errors) if errors else None,
                "mean_abs_error": sum(errors) / len(errors) if errors else None,
            }
    return result


def format_plan_scores(scores: Mapping) -> str:
    lines = [f"{'':<12}{'scale':>6}{'plans':>7}{'exact':>7}{'rate':>8}{'MAE':>7}"]
    for label, groups in scores.items():
        for group, row in groups.items():
            rate = f"{row['exact_rate']:.1%}" if row["exact_rate"] is not None else "-"
            mae = f"{row['mean_abs_error']:.2f}" if row["mean_abs_error"] is not None else "-"
            lines.append(f"{label:<12}{group:>6}{row['plans']:>7}{row['exact']:>7}{rate:>8}{mae:>7}")
    return "\n".join(lines)


# --- command line --------------------------------------------------------------------------


def _draw(labels: Mapping, stems: list[str], png_dir: Path, out_dir: Path) -> Iterator[Path]:
    from PIL import Image, ImageDraw

    colours = {"door": "red", "window": "blue", "toilet": "green", "washbasin": "magenta",
               "bathtub": "orange", "shower": "cyan", "washing machine": "brown"}
    out_dir.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        with Image.open(png_dir / f"{stem}.png") as picture:
            image = picture.convert("RGB")
        draw = ImageDraw.Draw(image)
        draw.rectangle(labels[stem]["area"], outline="grey", width=3)
        for box in plan_boxes(labels[stem]):
            draw.rectangle((box.left, box.top, box.right, box.bottom), outline=colours[box.label], width=2)
        path = out_dir / f"{stem}.png"
        image.save(path)
        yield path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m object_detector.swiss_dataset")
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("labels", help="DXF twins to boxes in PNG pixels, and their check")
    make.add_argument("--out", type=Path, required=True)
    make.add_argument("--png-dir", type=Path, default=PNG_DIR)
    make.add_argument("--dxf-dir", type=Path, default=DXF_DIR)
    make.add_argument("--questions", type=Path, default=QUESTIONS)
    draw = commands.add_parser("draw", help="draw the label boxes on PNG files")
    draw.add_argument("--labels", type=Path, required=True)
    draw.add_argument("--out", type=Path, required=True)
    draw.add_argument("--png-dir", type=Path, default=PNG_DIR)
    draw.add_argument("stems", nargs="+")
    args = parser.parse_args(argv)

    if args.command == "labels":
        labels = build_labels(args.png_dir, args.dxf_dir)
        answers = benchmark_answers(args.questions)
        # The answers go with the labels, so that the volume needs no copy of the benchmark.
        for stem, plan in labels.items():
            plan["answers"] = answers.get(stem, {})
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(labels))
        check = check_labels(labels, answers)
        args.out.with_name("label_check.json").write_text(json.dumps(check, indent=2))
        splits = Counter(plan["split"] for plan in labels.values())
        sites = {split: len({site_of(s) for s, p in labels.items() if p["split"] == split}) for split in splits}
        print("plans by split:", dict(splits), "sites by split:", sites)
        fits = np.array([plan["fit"] for plan in labels.values()])
        print(f"wall fit: min {fits.min():.3f}, 5th percentile {np.percentile(fits, 5):.3f}")
        print("boxes outside the drawing area:", sum(p["boxes_outside_area"] for p in labels.values()))
        for label, row in check.items():
            print(f"{label:<10} {row['exact']}/{row['plans']} plans exact; "
                  f"{row['unasked_with_label']} of {row['unasked_plans']} unasked plans have a label")
    else:
        labels = json.loads(args.labels.read_text())
        for path in _draw(labels, args.stems, args.png_dir, args.out):
            print(path)


if __name__ == "__main__":
    main()
