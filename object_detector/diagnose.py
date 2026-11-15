"""Where do the count errors of the detector come from? Box-level analysis on one split.

Usage (no GPU, no Modal):
    uv run python -m object_detector.diagnose <split_dir_with_coco> <detections.jsonl> [--overlays <images_dir> <out_dir>]

``split_dir_with_coco`` holds ``_annotations.coco.json`` of the split (valid). The detections
are the output of ``modal_predict.py``. The minimum scores are those of the object counter.
Prints recall and precision per class, the reasons of the missed labels and of the false
detections, and the exact-count rate by the true count.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from object_detector.classes import CLASSES

INPUT = 1024  # the model's input: the image is resized to a square of this side
MATCH_IOU = 0.5
LENIENT = False
"""With LENIENT, a detection also matches a label when one box lies inside the other (70%):
boxes drawn to another convention (a toilet with or without its tank) still count once."""


def same_object(a: list[float], b: list[float]) -> float:
    """A match score: the IoU, or with LENIENT also containment either way (as 0.5)."""
    value = iou(a, b)
    if LENIENT and value < MATCH_IOU and (inside(a, b) > 0.7 or inside(b, a) > 0.7):
        return MATCH_IOU
    return value


def iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def inside(a: list[float], b: list[float]) -> float:
    """The part of box ``a`` that is inside box ``b``."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area = (a[2] - a[0]) * (a[3] - a[1])
    return ix * iy / area if area > 0 else 0.0


def load_labels(split_dir: Path) -> tuple[dict[str, dict], dict[str, list[dict]]]:
    coco = json.loads((split_dir / "_annotations.coco.json").read_text())
    names = {c["id"]: c["name"] for c in coco["categories"]}
    images = {i["id"]: i for i in coco["images"]}
    labels: dict[str, list[dict]] = defaultdict(list)
    for a in coco["annotations"]:
        x, y, w, h = a["bbox"]
        labels[images[a["image_id"]]["file_name"]].append(
            {"label": names[a["category_id"]], "box": [x, y, x + w, y + h]}
        )
    return {i["file_name"]: i for i in coco["images"]}, labels


def match(gts: list[dict], dets: list[dict]) -> tuple[dict[int, int], list[int]]:
    """Greedy match by score at IoU >= 0.5. Returns gt index -> det index, and unmatched dets."""
    taken: dict[int, int] = {}
    free = []
    for d_index in sorted(range(len(dets)), key=lambda i: -dets[i]["score"]):
        best, best_iou = None, MATCH_IOU
        for g_index, gt in enumerate(gts):
            if g_index in taken:
                continue
            value = same_object(dets[d_index]["box"], gt["box"])
            if value >= best_iou:
                best, best_iou = g_index, value
        if best is None:
            free.append(d_index)
        else:
            taken[best] = d_index
    return taken, free


def size_bucket(box: list[float], width: int, height: int) -> str:
    """The shorter side of the box at the model's input, in pixels."""
    side = min((box[2] - box[0]) * INPUT / width, (box[3] - box[1]) * INPUT / height)
    for limit in (3, 6, 10, 20):
        if side < limit:
            return f"<{limit}px"
    return ">=20px"


def analyse(split_dir: Path, detections: Path, thresholds: dict[str, float], split: str) -> dict:
    images, labels = load_labels(split_dir)
    rows = {}
    for line in detections.open():
        row = json.loads(line)
        if row["image"] in images:
            rows[row["image"]] = row
    tp: Counter = Counter()
    fn: Counter = Counter()
    fp: Counter = Counter()
    miss_reason: dict[str, Counter] = defaultdict(Counter)
    fp_reason: dict[str, Counter] = defaultdict(Counter)
    by_size: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    by_count: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    plan_errors: dict[str, dict[str, int]] = {}
    for name, info in images.items():
        width, height = info["width"], info["height"]
        all_dets = rows.get(name, {"detections": []})["detections"]
        plan_errors[name] = {}
        for cls in CLASSES:
            gts = [g for g in labels[name] if g["label"] == cls]
            low = [d for d in all_dets if d["label"] == cls]
            kept = [d for d in low if d["score"] >= thresholds.get(cls, 0.5)]
            other = [d for d in all_dets if d["label"] != cls and d["score"] >= thresholds.get(d["label"], 0.5)]
            taken, free = match(gts, kept)
            tp[cls] += len(taken)
            fn[cls] += len(gts) - len(taken)
            fp[cls] += len(free)
            error = len(kept) - len(gts)
            plan_errors[name][cls] = error
            bucket = "0" if not gts else "1" if len(gts) == 1 else "2-3" if len(gts) <= 3 else "4-7" if len(gts) <= 7 else "8-15" if len(gts) <= 15 else "16+"
            by_count[cls][bucket]["plans"] += 1
            by_count[cls][bucket]["exact"] += error == 0
            for g_index, gt in enumerate(gts):
                size = size_bucket(gt["box"], width, height)
                by_size[cls][size]["labels"] += 1
                if g_index in taken:
                    continue
                by_size[cls][size]["missed"] += 1
                best_low = max((iou(d["box"], gt["box"]) for d in low), default=0.0)
                parts = [d for d in kept if inside(d["box"], gt["box"]) > 0.7]
                covering = [d for d in kept if inside(gt["box"], d["box"]) > 0.7]
                if any(iou(d["box"], gt["box"]) >= MATCH_IOU for d in kept):
                    reason = "its detection matched another label"
                elif best_low >= MATCH_IOU:
                    reason = "detected below the minimum score"
                elif len(parts) >= 1:
                    reason = "detected as smaller part(s)"
                elif covering:
                    reason = "inside a larger detection (merged)"
                elif any(iou(d["box"], gt["box"]) >= 0.5 for d in other):
                    reason = "detected as another class"
                elif best_low >= 0.1:
                    reason = "only a poorly placed box"
                else:
                    reason = "not detected at all"
                miss_reason[cls][reason] += 1
            for d_index in free:
                det = kept[d_index]
                if any(iou(det["box"], g["box"]) >= MATCH_IOU for g in gts):
                    reason = "duplicate of a matched label"
                elif any(inside(det["box"], g["box"]) > 0.7 for g in gts):
                    reason = "part of a label"
                elif any(inside(g["box"], det["box"]) > 0.7 for g in gts):
                    reason = "covers a label (merge)"
                elif any(iou(det["box"], g["box"]) >= 0.3 for g in labels[name] if g["label"] != cls):
                    reason = "on a label of another class"
                elif any(iou(det["box"], g["box"]) >= 0.1 for g in gts):
                    reason = "poorly placed on a label"
                else:
                    reason = "on no label"
                fp_reason[cls][reason] += 1
    return {
        "tp": tp, "fn": fn, "fp": fp, "miss_reason": miss_reason, "fp_reason": fp_reason,
        "by_size": by_size, "by_count": by_count, "plan_errors": plan_errors,
    }


def report(result: dict) -> None:
    print(f"{'class':<12}{'labels':>7}{'recall':>8}{'precision':>10}")
    for cls in CLASSES:
        n = result["tp"][cls] + result["fn"][cls]
        p = result["tp"][cls] + result["fp"][cls]
        print(f"{cls:<12}{n:>7}{result['tp'][cls] / max(n, 1):>8.1%}{result['tp'][cls] / max(p, 1):>10.1%}")
    for cls in CLASSES:
        print(f"\n== {cls}: missed {result['fn'][cls]}, false {result['fp'][cls]}")
        for reason, n in result["miss_reason"][cls].most_common():
            print(f"  missed: {reason:<40}{n:>6}")
        for reason, n in result["fp_reason"][cls].most_common():
            print(f"  false:  {reason:<40}{n:>6}")
        sizes = result["by_size"][cls]
        print("  shorter side at the 1024 input:", ", ".join(
            f"{s} {c['missed']}/{c['labels']} missed" for s, c in sorted(sizes.items())))
        counts = result["by_count"][cls]
        print("  exact count by true count:", ", ".join(
            f"{b}: {c['exact']}/{c['plans']}" for b, c in sorted(counts.items(), key=lambda x: x[0])))


def overlays(names: list[str], images_dir: Path, out_dir: Path, split_dir: Path, detections: Path,
             thresholds: dict[str, float], classes: tuple[str, ...]) -> None:
    from PIL import Image, ImageDraw

    Image.MAX_IMAGE_PIXELS = None
    _, labels = load_labels(split_dir)
    rows = {json.loads(l)["image"]: json.loads(l) for l in detections.open()}
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        with Image.open(images_dir / name) as picture:
            drawn = picture.convert("RGB")
        draw = ImageDraw.Draw(drawn)
        for gt in labels[name]:
            if gt["label"] in classes:
                draw.rectangle(gt["box"], outline=(0, 160, 0), width=3)
        for det in rows[name]["detections"]:
            if det["label"] in classes and det["score"] >= thresholds[det["label"]]:
                draw.rectangle(det["box"], outline=(220, 0, 0), width=2)
                draw.text((det["box"][0], det["box"][1] - 10), f"{det['score']:.2f}", fill=(220, 0, 0))
            elif det["label"] in classes and det["score"] >= 0.1:
                draw.rectangle(det["box"], outline=(0, 90, 255), width=1)
        drawn.save(out_dir / name)


def main(argv: list[str]) -> int:
    global LENIENT
    LENIENT = "--lenient" in argv
    argv = [a for a in argv if a != "--lenient"]
    from chatchtg.objects.counter import THRESHOLDS

    split_dir, detections = Path(argv[0]), Path(argv[1])
    result = analyse(split_dir, detections, THRESHOLDS, "valid")
    report(result)
    if "--overlays" in argv:
        i = argv.index("--overlays")
        images_dir, out_dir = Path(argv[i + 1]), Path(argv[i + 2])
        worst = sorted(result["plan_errors"], key=lambda n: -abs(result["plan_errors"][n]["window"]))[:12]
        worst += sorted(result["plan_errors"], key=lambda n: -abs(result["plan_errors"][n]["door"]))[:6]
        print("\noverlays:", worst)
        overlays(worst, images_dir, out_dir, split_dir, detections, THRESHOLDS, ("window", "door"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
