"""The Modal app that trains and scores the detector for the Swiss Dwellings plans.

It is a separate app from ``modal_app.py`` (the CubiCasa5K detector) so that its runs, logs
and ``modal app stop`` do not touch the other one. Both use the volume
``chatchtg-object-detector``. Run the commands from the root of the repository:

    uv run python -m object_detector.swiss_dataset labels --out var/swiss-detector/labels.json
    uv run --group train modal volume put chatchtg-object-detector \
        data/floorplans/data/swiss_dwellings raw/swiss-dwellings/png
    uv run --group train modal volume put chatchtg-object-detector \
        var/swiss-detector/labels.json datasets/swiss-v1/labels.json
    uv run --group train modal run -m object_detector.modal_swiss::build_tiles
    uv run --group train modal run --detach -m object_detector.modal_swiss::train \
        --run-name swiss-medium-640 --epochs 15
    uv run --group train modal run --detach -m object_detector.modal_swiss::train \
        --run-name swiss-medium-1280 --resolution 1280 --epochs 10 --batch-size 4 \
        --grad-accum-steps 4 --pretrained runs/swiss-medium-640/checkpoint_best_total.pth
    uv run --group train modal run -m object_detector.modal_swiss::evaluate --run-name swiss-medium-1280

See README.md and docs/benchmarks/2026-11-15-swiss-symbol-detector.md.
"""

from __future__ import annotations

import os

import modal

APP_NAME = "chatchtg-swiss-detector"
VOLUME_NAME = "chatchtg-object-detector"
VOLUME_DIR = "/vol"
PNG_DIR = f"{VOLUME_DIR}/raw/swiss-dwellings/png"

_GPU = os.environ.get("OBJECT_DETECTOR_GPU", "L40S")
TRAIN_GPU = None if _GPU == "none" else _GPU
PREDICT_GPU = None if _GPU == "none" else os.environ.get("OBJECT_DETECTOR_PREDICT_GPU", "A10G")

MODEL_CLASSES = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "large": "RFDETRLarge",
}
SCORE_FLOOR = 0.05
"""Detections down to this score are stored, so that a threshold can be chosen later."""

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False, version=2)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")
    .uv_pip_install("rfdetr[train]==1.12.1")
    .add_local_python_source("object_detector")
)


def _dataset_dir(dataset: str) -> str:
    return f"{VOLUME_DIR}/datasets/{dataset}"


def _run_dir(run_name: str) -> str:
    if not run_name.startswith("swiss-"):
        raise ValueError("The name of a Swiss run starts with 'swiss-', to keep it apart from other runs.")
    return f"{VOLUME_DIR}/runs/{run_name}"


@app.function(image=image, volumes={VOLUME_DIR: volume}, timeout=2 * 60 * 60, cpu=16, memory=16384)
def build_tiles(dataset: str = "swiss-v1") -> None:
    """Cut the sheets into COCO tiles, with the labels of ``datasets/<dataset>/labels.json``."""
    import json
    import shutil
    from pathlib import Path

    from object_detector import swiss_dataset

    if not dataset.startswith("swiss-"):
        raise ValueError("A Swiss dataset name starts with 'swiss-'.")
    out_dir = Path(_dataset_dir(dataset))
    labels = json.loads((out_dir / "labels.json").read_text())
    for split in ("train", "valid", "test"):
        if (out_dir / split).exists():
            shutil.rmtree(out_dir / split)
    stats = swiss_dataset.coco_tiles(labels, Path(PNG_DIR), out_dir)
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    volume.commit()


@app.function(
    image=image,
    volumes={VOLUME_DIR: volume},
    gpu=TRAIN_GPU,
    cpu=8,
    memory=32768,
    timeout=24 * 60 * 60,
    retries=modal.Retries(max_retries=3, initial_delay=0.0),
)
def train(
    run_name: str,
    size: str = "medium",
    resolution: int = 640,
    epochs: int = 12,
    batch_size: int = 8,
    grad_accum_steps: int = 2,
    dataset: str = "swiss-v1",
    limit: int = 0,
    pretrained: str = "",
) -> None:
    """Fine-tune RF-DETR on the tiles. The run is stored in runs/<run_name> on the volume.

    ``pretrained`` is a checkpoint on the volume to start from, for example
    ``runs/large-1024/checkpoint_best_total.pth`` (read only). Without it the run starts
    from the COCO weights of RF-DETR. ``limit`` trains on the first tiles of each split only.
    """
    import json
    import time
    from pathlib import Path

    import rfdetr

    from object_detector import dataset as datasets

    if resolution % 32:
        raise ValueError("The resolution must be a multiple of 32.")
    model_class = getattr(rfdetr, MODEL_CLASSES[size])
    dataset_dir = Path(_dataset_dir(dataset))
    if limit:
        datasets.subset_dataset(dataset_dir, Path("/tmp/subset"), limit)
        dataset_dir = Path("/tmp/subset")
    run_dir = Path(_run_dir(run_name))
    run_dir.mkdir(parents=True, exist_ok=True)
    settings = {
        "size": size, "resolution": resolution, "epochs": epochs, "batch_size": batch_size,
        "grad_accum_steps": grad_accum_steps, "dataset": dataset, "limit": limit,
        "pretrained": pretrained, "gpu": TRAIN_GPU,
    }
    (run_dir / "run_settings.json").write_text(json.dumps(settings, indent=2))
    checkpoint = run_dir / "last.ckpt"
    print("resuming from", checkpoint if checkpoint.exists() else pretrained or "the COCO weights", flush=True)

    start = time.time()
    model = model_class(pretrain_weights=f"{VOLUME_DIR}/{pretrained}") if pretrained else model_class()
    model.train(
        dataset_dir=str(dataset_dir),
        output_dir=str(run_dir),
        epochs=epochs,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        resolution=resolution,
        resume=str(checkpoint) if checkpoint.exists() else None,
        early_stopping=True,
        early_stopping_patience=5,
        tensorboard=True,
    )
    minutes = (time.time() - start) / 60
    print(f"training took {minutes:.1f} minutes", flush=True)
    (run_dir / "train_minutes.txt").write_text(f"{minutes:.1f}\n")
    volume.commit()


@app.function(image=image, volumes={VOLUME_DIR: volume}, gpu=PREDICT_GPU, cpu=8, memory=16384,
              timeout=4 * 60 * 60)
def evaluate(run_name: str, dataset: str = "swiss-v1", limit: int = 0, batch: int = 16) -> None:
    """Find the objects on every Swiss sheet, choose a threshold per class on the valid
    split, and score the test split with those thresholds.

    Writes to runs/<run_name>/:
    - ``detections_all.json``: for each plan, its split, its scale and every detection with a
      score of at least SCORE_FLOOR (class, score, box in PNG pixels);
    - ``scores.json``: the thresholds and the scores of the valid and test splits;
    - ``samples/``: a few test sheets with the kept detections drawn.
    """
    import json
    import time
    from collections import Counter
    from pathlib import Path

    import numpy as np
    from PIL import Image, ImageDraw
    from rfdetr import RFDETR

    from object_detector import swiss_dataset as sd
    from object_detector.swiss_dxf import ASKED_CLASSES, SWISS_CLASSES

    volume.reload()
    run_dir = Path(_run_dir(run_name))
    labels = json.loads((Path(_dataset_dir(dataset)) / "labels.json").read_text())
    stems = sorted(labels)
    if limit:
        # A few plans of each split, for a quick check.
        stems = [s for split in ("train", "valid", "test")
                 for s in [s for s in stems if labels[s]["split"] == split][:limit]]
    settings = json.loads((run_dir / "run_settings.json").read_text())
    model = RFDETR.from_checkpoint(str(run_dir / "checkpoint_best_total.pth"))
    print("classes of the checkpoint:", list(model.class_names), flush=True)

    start = time.time()
    detections: dict[str, dict] = {}
    for index, stem in enumerate(stems):
        with Image.open(Path(PNG_DIR) / f"{stem}.png") as picture:
            sheet = picture.convert("RGB")
        grey = np.asarray(sheet.convert("L"))
        # The drawing area comes from the PNG only; the legend under it is left out.
        area = sd.drawing_area(grey)
        corners = sd.sheet_tiles(grey, area)
        found = []
        for first in range(0, len(corners), batch):
            chunk = corners[first:first + batch]
            crops = [sheet.crop((x, y, x + sd.TILE, y + sd.TILE)) for x, y in chunk]
            results = model.predict(crops, threshold=SCORE_FLOOR)
            if not isinstance(results, list):
                results = [results]
            for corner, result in zip(chunk, results):
                found.append((corner, [
                    {"label": str(label), "score": round(float(score), 4),
                     "box": [round(float(v), 1) for v in box]}
                    for box, score, label in zip(result.xyxy, result.confidence, result.data["class_name"])
                    if str(label) in SWISS_CLASSES
                ]))
        detections[stem] = {
            "split": labels[stem]["split"],
            "scale": labels[stem]["scale"],
            "area": list(area),
            "detections": sd.join_tiles(found, area),
        }
        if index % 50 == 0:
            print(f"{index} of {len(stems)} sheets, {time.time() - start:.0f} s", flush=True)
    predict_seconds = time.time() - start

    def counts(split: str, thresholds: dict[str, float]) -> dict[str, Counter]:
        return {
            s: Counter(d["label"] for d in row["detections"] if d["score"] >= thresholds[d["label"]])
            for s, row in detections.items() if row["split"] == split
        }

    # The answer of every plan, also of the plans without a bathtub or shower question
    # (their count is 0, checked against the DXF labels).
    truth = {s: sd.counts_of(labels[s]) for s in detections}
    grid = [round(v, 2) for v in np.arange(0.1, 0.96, 0.05)]
    thresholds = {name: 0.5 for name in SWISS_CLASSES}
    valid_choice = {}
    for name in ASKED_CLASSES:
        rows = []
        for value in grid:
            found_counts = counts("valid", {**thresholds, name: value})
            errors = [abs(found_counts[s][name] - truth[s][name]) for s in found_counts]
            if errors:
                rows.append((sum(e == 0 for e in errors) / len(errors), -sum(errors) / len(errors), value))
        best = max(rows) if rows else (None, None, 0.5)
        thresholds[name] = best[2]
        valid_choice[name] = {"threshold": best[2], "exact_rate_all_valid_plans": best[0]}
    print("thresholds chosen on the valid split:", thresholds, flush=True)

    answers = {s: labels[s]["answers"] for s in labels}
    result = {
        "run": run_name,
        "settings": settings,
        "thresholds": thresholds,
        "valid_choice": valid_choice,
        "predict_seconds": round(predict_seconds),
        "sheets": len(detections),
    }
    for split in ("valid", "test"):
        split_stems = [s for s in detections if detections[s]["split"] == split]
        result[split] = sd.score_plans(counts(split, thresholds), labels, answers, split_stems)
        print(f"{split} split, counts against the benchmark answers:")
        print(sd.format_plan_scores(result[split]), flush=True)

    (run_dir / "detections_all.json").write_text(json.dumps({
        "classes": SWISS_CLASSES, "score_floor": SCORE_FLOOR, "thresholds": thresholds,
        "box": "x0, y0, x1, y1 in pixels of the PNG", "plans": detections,
    }))
    (run_dir / "scores.json").write_text(json.dumps(result, indent=2))
    samples = run_dir / "samples"
    samples.mkdir(exist_ok=True)
    colours = {"door": "red", "window": "blue", "toilet": "green", "washbasin": "magenta",
               "bathtub": "orange", "shower": "cyan", "washing machine": "brown"}
    for stem in [s for s in detections if detections[s]["split"] == "test"][:6]:
        with Image.open(Path(PNG_DIR) / f"{stem}.png") as picture:
            drawn = picture.convert("RGB")
        draw = ImageDraw.Draw(drawn)
        for d in detections[stem]["detections"]:
            if d["score"] >= thresholds[d["label"]]:
                draw.rectangle(d["box"], outline=colours[d["label"]], width=2)
        drawn.save(samples / f"{stem}.png")
    volume.commit()
