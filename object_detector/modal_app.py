"""The Modal app that trains the detector. Run the commands from the root of the repository:

    uv run --group train modal run -m object_detector.modal_app::smoke
    uv run --group train modal run -m object_detector.modal_app::download
    uv run --group train modal run -m object_detector.modal_app::build_dataset
    uv run --group train modal run --detach -m object_detector.modal_app::train --run-name large-1024
    uv run --group train modal run -m object_detector.modal_app::evaluate --run-name large-1024

Everything is stored on the Modal volume ``chatchtg-object-detector``. See README.md.
"""

from __future__ import annotations

import modal

APP_NAME = "chatchtg-object-detector"
VOLUME_DIR = "/vol"
RAW_DIR = f"{VOLUME_DIR}/raw/cubicasa5k"
BENCHMARK_QUESTIONS = f"{VOLUME_DIR}/benchmark/questions.jsonl"

# CubiCasa5K, https://zenodo.org/records/2613548, CC BY-NC-SA 4.0.
ZIP_URL = "https://zenodo.org/api/records/2613548/files/cubicasa5k.zip/content"
ZIP_MD5 = "0ce0b203d1e3c125b51087b219bd23b9"

# RF-DETR class name of each size. The sizes after "large" are not Apache 2.0: they need
# the rfdetr[plus] package and Roboflow's PML licence.
MODEL_CLASSES = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "large": "RFDETRLarge",
    "xlarge": "RFDETRXLarge",
    "2xlarge": "RFDETR2XLarge",
}

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(APP_NAME, create_if_missing=True, version=2)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")
    .uv_pip_install("rfdetr[train]==1.12.1")
    .add_local_python_source("object_detector")
)


def _dataset_dir(dataset: str) -> str:
    return f"{VOLUME_DIR}/datasets/{dataset}"


def _run_dir(run_name: str) -> str:
    return f"{VOLUME_DIR}/runs/{run_name}"


@app.function(image=image, gpu="L4", timeout=600)
def smoke() -> None:
    """Show that the account, the image and a GPU work."""
    import rfdetr
    import torch

    print("torch", torch.__version__, "| CUDA:", torch.cuda.is_available())
    print("GPU:", torch.cuda.get_device_name(0))
    print("rfdetr", getattr(rfdetr, "__version__", "?"))
    print("models:", sorted(name for name in dir(rfdetr) if name.startswith("RFDETR")))


@app.function(image=image, volumes={VOLUME_DIR: volume}, timeout=2 * 60 * 60, cpu=2)
def download() -> None:
    """Download CubiCasa5K (5.5 GB) and extract it to the volume. Does nothing if it is there."""
    import hashlib
    import urllib.request
    import zipfile
    from pathlib import Path

    done = Path(RAW_DIR) / ".complete"
    if done.exists():
        print("CubiCasa5K is on the volume already.")
        return
    archive = Path("/tmp/cubicasa5k.zip")
    digest = hashlib.md5()
    with urllib.request.urlopen(ZIP_URL, timeout=120) as response, archive.open("wb") as file:
        size = 0
        while chunk := response.read(1 << 22):
            file.write(chunk)
            digest.update(chunk)
            if (size + len(chunk)) >> 30 != size >> 30:
                print(f"downloaded {(size + len(chunk)) >> 30} GB", flush=True)
            size += len(chunk)
    if digest.hexdigest() != ZIP_MD5:
        raise RuntimeError(f"The archive has MD5 {digest.hexdigest()}, expected {ZIP_MD5}.")
    print("checksum is correct, extracting", flush=True)
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(Path(RAW_DIR).parent)
    done.touch()
    volume.commit()
    print("plans:", sum(1 for _ in Path(RAW_DIR).glob("*/*/model.svg")))


@app.function(image=image, volumes={VOLUME_DIR: volume}, timeout=60 * 60, cpu=2)
def build_dataset(dataset: str = "cubicasa-v1", limit: int = 0) -> None:
    """Convert the labels to a COCO dataset on the volume and print its statistics."""
    import json
    import shutil
    from pathlib import Path

    from object_detector import dataset as datasets
    from object_detector import scoring

    out_dir = Path(_dataset_dir(dataset))
    if out_dir.exists():
        shutil.rmtree(out_dir)
    stats = datasets.build_dataset(Path(RAW_DIR), out_dir, limit=limit or None)
    for split, row in stats["splits"].items():
        print(f"{split}: {row['images']} of {row['plans']} plans, boxes {row['boxes']}")
        print(f"  boxes outside the image, dropped: {row['dropped_boxes']}")
        for plan, reason in row["skipped_plans"].items():
            print(f"  skipped {plan}: {reason}")
    if Path(BENCHMARK_QUESTIONS).exists():
        answers = scoring.benchmark_answers(Path(BENCHMARK_QUESTIONS))
        counts = {}
        for split in stats["splits"]:
            counts.update(datasets.label_counts(out_dir / split))
        check = scoring.score_against_benchmark(counts, answers)
        (out_dir / "label_check.json").write_text(json.dumps(check, indent=2))
        print("Label counts against the answers of the benchmark:")
        print(scoring.format_scores(check))
    else:
        print(f"No {BENCHMARK_QUESTIONS}: the labels are not compared with the benchmark.")
    volume.commit()


@app.function(
    image=image,
    volumes={VOLUME_DIR: volume},
    gpu="L40S",
    cpu=8,
    memory=32768,
    timeout=24 * 60 * 60,
    # A GPU container can be stopped at any time. The next attempt resumes from last.ckpt.
    retries=modal.Retries(max_retries=5, initial_delay=0.0),
)
def train(
    run_name: str,
    size: str = "large",
    resolution: int = 1024,
    epochs: int = 40,
    batch_size: int = 4,
    grad_accum_steps: int = 4,
    dataset: str = "cubicasa-v1",
    limit: int = 0,
) -> None:
    """Fine-tune RF-DETR on the dataset. The run is stored in runs/<run_name> on the volume.

    ``limit`` trains on the first images of each split only, for a quick check.
    """
    import json
    import time
    from pathlib import Path

    import rfdetr

    from object_detector import dataset as datasets

    if resolution % 32:
        raise ValueError("The resolution must be a multiple of 32.")
    model_class = getattr(rfdetr, MODEL_CLASSES[size], None)
    if model_class is None:
        raise ValueError(f"This image has no {MODEL_CLASSES[size]}. It needs rfdetr[plus].")
    dataset_dir = Path(_dataset_dir(dataset))
    if limit:
        datasets.subset_dataset(dataset_dir, Path("/tmp/subset"), limit)
        dataset_dir = Path("/tmp/subset")
    run_dir = Path(_run_dir(run_name))
    run_dir.mkdir(parents=True, exist_ok=True)
    settings = {
        "size": size,
        "resolution": resolution,
        "epochs": epochs,
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum_steps,
        "dataset": dataset,
        "limit": limit,
    }
    (run_dir / "run_settings.json").write_text(json.dumps(settings, indent=2))
    checkpoint = run_dir / "last.ckpt"
    print("resuming from", checkpoint if checkpoint.exists() else "the pretrained weights", flush=True)

    start = time.time()
    model = model_class()
    model.train(
        dataset_dir=str(dataset_dir),
        output_dir=str(run_dir),
        epochs=epochs,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        resolution=resolution,
        resume=str(checkpoint) if checkpoint.exists() else None,
        early_stopping=True,
        early_stopping_patience=10,
        tensorboard=True,
    )
    print(f"training took {(time.time() - start) / 60:.1f} minutes", flush=True)
    volume.commit()


@app.function(image=image, volumes={VOLUME_DIR: volume}, gpu="L4", timeout=2 * 60 * 60)
def evaluate(run_name: str, dataset: str = "cubicasa-v1", limit: int = 0) -> None:
    """Score the best checkpoint of a run on the test split by its counts of objects.

    The minimum score of a detection is the one that gives the best counts on the
    validation split. The test split is scored with that value only.
    """
    import json
    from collections import Counter
    from pathlib import Path

    from PIL import Image, ImageDraw
    from rfdetr import RFDETR

    from object_detector import dataset as datasets
    from object_detector import scoring
    from object_detector.classes import CLASSES

    volume.reload()
    run_dir = Path(_run_dir(run_name))
    dataset_dir = Path(_dataset_dir(dataset))
    model = RFDETR.from_checkpoint(str(run_dir / "checkpoint_best_total.pth"))
    names = list(model.class_names)
    print("classes of the checkpoint:", names)

    def predict(split: str) -> dict[str, list[dict]]:
        """Image name -> detections with a score of at least 0.1."""
        coco = json.loads((dataset_dir / split / "_annotations.coco.json").read_text())
        images = coco["images"][: limit or None]
        found = {}
        for index, row in enumerate(images):
            with Image.open(dataset_dir / split / row["file_name"]) as picture:
                detections = model.predict(picture.convert("RGB"), threshold=0.1)
            found[row["file_name"]] = [
                {"label": names[int(class_id)], "score": float(score), "box": [float(v) for v in box]}
                for box, score, class_id in zip(
                    detections.xyxy, detections.confidence, detections.class_id
                )
            ]
            if index % 50 == 0:
                print(f"{split}: {index} of {len(images)}", flush=True)
        return found

    def counts(found: dict[str, list[dict]], threshold: float) -> dict[str, Counter]:
        return {
            image: Counter(d["label"] for d in detections if d["score"] >= threshold)
            for image, detections in found.items()
        }

    def labels(split: str, found: dict[str, list[dict]]) -> dict[str, Counter]:
        return {k: v for k, v in datasets.label_counts(dataset_dir / split).items() if k in found}

    valid = predict("valid")
    valid_labels = labels("valid", valid)
    by_threshold = {}
    for step in range(2, 19):
        threshold = step / 20
        scores = scoring.score_counts(counts(valid, threshold), valid_labels, CLASSES)
        by_threshold[threshold] = sum(row["exact_rate"] for row in scores.values()) / len(scores)
    threshold = max(by_threshold, key=by_threshold.get)
    print(f"minimum score {threshold:.2f} gives the best counts on the validation split")

    test = predict("test")
    test_counts = counts(test, threshold)
    result = {
        "threshold": threshold,
        "valid_mean_exact_rate_by_threshold": by_threshold,
        "test_images": len(test),
        "against_labels": scoring.score_counts(test_counts, labels("test", test), CLASSES),
    }
    print("Test split, counts against the labels:")
    print(scoring.format_scores(result["against_labels"]))
    if Path(BENCHMARK_QUESTIONS).exists():
        answers = scoring.benchmark_answers(Path(BENCHMARK_QUESTIONS))
        result["against_benchmark"] = scoring.score_against_benchmark(test_counts, answers)
        print("Test split, counts against the answers of the benchmark:")
        print(scoring.format_scores(result["against_benchmark"]))

    (run_dir / "predictions_test.json").write_text(json.dumps(test))
    (run_dir / "scores.json").write_text(json.dumps(result, indent=2))
    samples = run_dir / "samples"
    samples.mkdir(exist_ok=True)
    for name in list(test)[:8]:
        with Image.open(dataset_dir / "test" / name) as picture:
            drawn = picture.convert("RGB")
        draw = ImageDraw.Draw(drawn)
        for detection in test[name]:
            if detection["score"] >= threshold:
                draw.rectangle(detection["box"], outline="red", width=2)
                draw.text(detection["box"][:2], f"{detection['label']} {detection['score']:.2f}", fill="red")
        drawn.save(samples / name)
    volume.commit()
