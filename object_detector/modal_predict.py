"""Run a trained detector over every plan of CubiCasa5K and store all its detections.

    uv run --group train modal run -m object_detector.modal_predict::predict_all --run-name large-1024
    uv run --group train modal volume get chatchtg-object-detector predictions/large-1024/detections.jsonl var/detections/

The plans are read from the extracted dataset (``raw/cubicasa5k``), in the split of
CubiCasa5K, so the plans that ``build_dataset`` skipped are predicted too. Each line of
``predictions/<run>/detections.jsonl`` is one plan::

    {"image": "colorful-10052.png", "plan": "/colorful/10052/", "split": "train",
     "in_dataset": true, "width": 1240, "height": 877, "sha256": "...",
     "detections": [{"label": "door", "score": 0.93, "box": [x0, y0, x1, y1]}, ...]}

Boxes are in pixels of ``F1_original.png``, the image of the benchmark. Detections down to
``--min-score`` (default 0.05) are kept, so that the minimum score of a count can be chosen later.

This uses the image and the volume of ``modal_app.py`` but its own app name, so it does not
mix with a training or evaluation run of that app.
"""

from __future__ import annotations

import os

import modal

from object_detector.modal_app import RAW_DIR, VOLUME_DIR, image, volume

APP_NAME = "chatchtg-object-predict"
# Set OBJECT_DETECTOR_PREDICT_GPU to use another type, for example T4, A10G or L40S.
PREDICT_GPU = os.environ.get("OBJECT_DETECTOR_PREDICT_GPU", "L4")
CHUNK = 250

app = modal.App(APP_NAME)


def _out_dir(run_name: str) -> str:
    return f"{VOLUME_DIR}/predictions/{run_name}"


@app.function(image=image, volumes={VOLUME_DIR: volume}, timeout=10 * 60)
def list_plans(dataset: str = "cubicasa-v1") -> list[tuple[str, str, bool]]:
    """(split, plan, in_dataset) for every plan of the split files of CubiCasa5K."""
    import json
    from pathlib import Path

    from object_detector.dataset import SPLITS, image_name

    in_dataset: set[str] = set()
    for split in SPLITS.values():
        coco = Path(f"{VOLUME_DIR}/datasets/{dataset}/{split}/_annotations.coco.json")
        if coco.exists():
            in_dataset.update(row["file_name"] for row in json.loads(coco.read_text())["images"])
    plans = []
    for file_name, split in SPLITS.items():
        for line in (Path(RAW_DIR) / file_name).read_text().split():
            plans.append((split, line, image_name(line) in in_dataset))
    return plans


@app.function(
    image=image,
    volumes={VOLUME_DIR: volume},
    gpu=PREDICT_GPU,
    timeout=60 * 60,
    retries=modal.Retries(max_retries=2, initial_delay=0.0),
)
def predict_chunk(
    index: int,
    plans: list[tuple[str, str, bool]],
    run_name: str,
    checkpoint: str,
    min_score: float,
    out_name: str,
) -> int:
    """Predict one chunk of plans with ``runs/<run_name>/<checkpoint>`` and write it to
    ``predictions/<out_name>/chunks/<index>.jsonl``. Returns the number of plans."""
    import hashlib
    import json
    import time
    from pathlib import Path

    from PIL import Image
    from rfdetr import RFDETR

    from object_detector.classes import CLASSES
    from object_detector.dataset import image_name

    Image.MAX_IMAGE_PIXELS = None
    target = Path(_out_dir(out_name)) / "chunks" / f"{index:04d}.jsonl"
    if target.exists():
        return len(plans)
    model = RFDETR.from_checkpoint(f"{VOLUME_DIR}/runs/{run_name}/{checkpoint}")
    start = time.time()
    lines = []
    for split, plan, in_dataset in plans:
        path = Path(RAW_DIR) / plan.strip("/") / "F1_original.png"
        content = path.read_bytes()
        with Image.open(path) as picture:
            width, height = picture.size
            detections = model.predict(picture.convert("RGB"), threshold=min_score)
        found = [
            {
                "label": str(label),
                "score": round(float(score), 4),
                "box": [round(float(v), 1) for v in box],
            }
            for box, score, label in zip(
                detections.xyxy, detections.confidence, detections.data["class_name"]
            )
            if str(label) in CLASSES
        ]
        lines.append(
            json.dumps(
                {
                    "image": image_name(plan),
                    "plan": plan,
                    "split": split,
                    "in_dataset": in_dataset,
                    "width": width,
                    "height": height,
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "detections": found,
                }
            )
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    tmp.write_text("\n".join(lines) + "\n")
    tmp.replace(target)
    volume.commit()
    print(f"chunk {index}: {len(plans)} plans in {time.time() - start:.0f} s", flush=True)
    return len(plans)


@app.function(image=image, volumes={VOLUME_DIR: volume}, timeout=20 * 60)
def merge(run_name: str, chunks: int, settings: dict) -> str:
    """Join the chunks into ``detections.jsonl`` and write ``settings.json``."""
    import json
    from pathlib import Path

    volume.reload()
    out_dir = Path(_out_dir(run_name))
    parts = [out_dir / "chunks" / f"{index:04d}.jsonl" for index in range(chunks)]
    missing = [part.name for part in parts if not part.exists()]
    if missing:
        raise RuntimeError(f"Chunks are missing: {missing}")
    tmp = out_dir / "detections.jsonl.tmp"
    with tmp.open("w") as out:
        for part in parts:
            out.write(part.read_text())
    tmp.replace(out_dir / "detections.jsonl")
    (out_dir / "settings.json").write_text(json.dumps(settings, indent=2))
    volume.commit()
    return str(out_dir / "detections.jsonl")


@app.local_entrypoint()
def predict_all(
    run_name: str = "large-1024",
    checkpoint: str = "checkpoint_best_total.pth",
    min_score: float = 0.05,
    limit: int = 0,
    out_name: str = "",
) -> None:
    """Predict every plan of CubiCasa5K with a checkpoint of ``runs/<run_name>``.

    The output goes to ``predictions/<out_name>``, by default ``predictions/<run_name>``.
    ``limit`` predicts the first plans only, for a quick check; its output goes to
    ``predictions/<run_name>-limit<limit>``.
    """
    import time

    plans = list_plans.remote()
    if limit:
        plans = plans[:limit]
    out_name = out_name or (f"{run_name}-limit{limit}" if limit else run_name)
    chunks = [plans[i : i + CHUNK] for i in range(0, len(plans), CHUNK)]
    print(f"{len(plans)} plans in {len(chunks)} chunks on {PREDICT_GPU}")
    start = time.time()
    jobs = [(i, chunk, run_name, checkpoint, min_score, out_name) for i, chunk in enumerate(chunks)]
    done = 0
    for count in predict_chunk.starmap(jobs):
        done += count
        print(f"{done} of {len(plans)} plans, {time.time() - start:.0f} s", flush=True)
    settings = {
        "run_name": run_name,
        "checkpoint": checkpoint,
        "min_score": min_score,
        "gpu": PREDICT_GPU,
        "plans": len(plans),
        "seconds": round(time.time() - start),
    }
    print("written:", merge.remote(out_name, len(chunks), settings))
