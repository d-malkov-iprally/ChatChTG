"""Build a COCO dataset for the detector from an extracted CubiCasa5K folder.

Layout of the result, as RF-DETR reads it::

    <out>/train/_annotations.coco.json   and the images of the split next to it
    <out>/valid/...
    <out>/test/...
    <out>/stats.json

The images are ``F1_original.png`` of each plan under the name ``<category>-<id>.png``.
That is the name and the size of the same plan in the benchmark of this project.
"""

from __future__ import annotations

import json
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from object_detector.classes import CLASSES
from object_detector.cubicasa_svg import Box, read_boxes

# Some sheets are larger than the size at which Pillow warns of a decompression bomb.
Image.MAX_IMAGE_PIXELS = None

# CubiCasa5K split file -> folder name of RF-DETR.
SPLITS = {"train.txt": "train", "val.txt": "valid", "test.txt": "test"}
# A box that is thinner than this after it is cut to the image is dropped.
MIN_SIDE = 1.0
# A plan is dropped if more than this part of its boxes is not inside the image: the
# labels do not belong to the image then (CubiCasa5K issues 20 and 71).
MAX_OUTSIDE = 0.2


def image_name(plan: str) -> str:
    """ "/high_quality/5731/" -> "high_quality-5731.png"."""
    category, identifier = plan.strip("/").split("/")
    return f"{category}-{identifier}.png"


def plan_boxes(plan_dir: Path) -> tuple[tuple[int, int], list[Box], int]:
    """The size of ``F1_original.png``, the boxes in its pixels, and the number of dropped boxes."""
    with Image.open(plan_dir / "F1_original.png") as image:
        width, height = image.size
    with Image.open(plan_dir / "F1_scaled.png") as image:
        scaled_width, scaled_height = image.size
    # The labels are in the pixels of F1_scaled.png.
    scale_x, scale_y = width / scaled_width, height / scaled_height
    boxes = []
    dropped = 0
    for box in read_boxes((plan_dir / "model.svg").read_text(encoding="utf-8")):
        left = min(max(box.left * scale_x, 0.0), float(width))
        right = min(max(box.right * scale_x, 0.0), float(width))
        top = min(max(box.top * scale_y, 0.0), float(height))
        bottom = min(max(box.bottom * scale_y, 0.0), float(height))
        if right - left < MIN_SIDE or bottom - top < MIN_SIDE:
            dropped += 1
        else:
            boxes.append(Box(box.label, left, top, right, bottom))
    return (width, height), boxes, dropped


def _convert_plan(raw_dir: Path, split_dir: Path, plan: str) -> dict:
    """Read the labels of one plan and copy its image. Returns the plan's row, or the reason
    why the plan is skipped."""
    plan_dir = raw_dir / plan.strip("/")
    try:
        (width, height), boxes, dropped = plan_boxes(plan_dir)
    except (OSError, SyntaxError, ValueError) as error:
        return {"skipped": f"not readable: {error}"}
    if dropped > MAX_OUTSIDE * (len(boxes) + dropped):
        return {"skipped": f"{dropped} of {len(boxes) + dropped} boxes are outside the image"}
    name = image_name(plan)
    shutil.copyfile(plan_dir / "F1_original.png", split_dir / name)
    return {"name": name, "width": width, "height": height, "boxes": boxes, "dropped": dropped}


def build_dataset(raw_dir: Path, out_dir: Path, limit: int | None = None, workers: int = 32) -> dict:
    """Write the dataset to ``out_dir`` and return its statistics.

    ``raw_dir`` is the extracted ``cubicasa5k`` folder. ``limit`` takes only the first
    plans of each split, for a quick check. The plans are read by ``workers`` threads:
    on a network volume, the time goes to reading and copying files.
    """
    stats: dict = {"classes": CLASSES, "splits": {}}
    for split_file, split in SPLITS.items():
        plans = (raw_dir / split_file).read_text().split()[:limit]
        split_dir = out_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            # map keeps the order of the plans, so the ids are the same in every build.
            rows = list(pool.map(lambda plan: _convert_plan(raw_dir, split_dir, plan), plans))
        images, annotations = [], []
        boxes_by_class: Counter[str] = Counter()
        skipped: dict[str, str] = {}
        dropped_boxes = 0
        for plan, row in zip(plans, rows):
            if "skipped" in row:
                skipped[plan] = row["skipped"]
                continue
            dropped_boxes += row["dropped"]
            image_id = len(images)
            images.append(
                {"id": image_id, "file_name": row["name"], "width": row["width"], "height": row["height"]}
            )
            for box in row["boxes"]:
                box_width, box_height = box.right - box.left, box.bottom - box.top
                annotations.append(
                    {
                        "id": len(annotations),
                        "image_id": image_id,
                        "category_id": CLASSES.index(box.label),
                        "bbox": [round(v, 2) for v in (box.left, box.top, box_width, box_height)],
                        "area": round(box_width * box_height, 2),
                        "iscrowd": 0,
                    }
                )
                boxes_by_class[box.label] += 1
        categories = [
            {"id": index, "name": name, "supercategory": "symbol"}
            for index, name in enumerate(CLASSES)
        ]
        coco = {"images": images, "annotations": annotations, "categories": categories}
        (split_dir / "_annotations.coco.json").write_text(json.dumps(coco))
        stats["splits"][split] = {
            "plans": len(plans),
            "images": len(images),
            "boxes": {name: boxes_by_class[name] for name in CLASSES},
            "dropped_boxes": dropped_boxes,
            "skipped_plans": skipped,
        }
        print(f"{split}: {len(images)} images written", flush=True)
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def label_counts(split_dir: Path) -> dict[str, Counter[str]]:
    """Image name -> number of labelled boxes of each class, for one split of a built dataset."""
    coco = json.loads((split_dir / "_annotations.coco.json").read_text())
    names = {image["id"]: image["file_name"] for image in coco["images"]}
    counts: dict[str, Counter[str]] = {name: Counter() for name in names.values()}
    for annotation in coco["annotations"]:
        counts[names[annotation["image_id"]]][CLASSES[annotation["category_id"]]] += 1
    return counts


def subset_dataset(dataset_dir: Path, out_dir: Path, limit: int) -> None:
    """Write a dataset of the first ``limit`` images of each split, for a quick check.

    The images are links to the images of ``dataset_dir``.
    """
    for split in SPLITS.values():
        coco = json.loads((dataset_dir / split / "_annotations.coco.json").read_text())
        images = coco["images"][:limit]
        kept = {image["id"] for image in images}
        coco["images"] = images
        coco["annotations"] = [a for a in coco["annotations"] if a["image_id"] in kept]
        (out_dir / split).mkdir(parents=True, exist_ok=True)
        (out_dir / split / "_annotations.coco.json").write_text(json.dumps(coco))
        for image in images:
            link = out_dir / split / image["file_name"]
            link.unlink(missing_ok=True)
            link.symlink_to((dataset_dir / split / image["file_name"]).resolve())
