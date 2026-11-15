"""Continue training a finished run at another input resolution, in its own Modal app.

    uv run --group train modal run -m object_detector.modal_finetune::finetune \
        --run-name large-1536-ft --init-run large-1024 --resolution 1536 --epochs 12
    uv run --group train modal run --detach -m object_detector.modal_finetune::finetune ...

The weights of ``runs/<init_run>/checkpoint_best_total.pth`` are the start; the new run is
written to ``runs/<run_name>`` and continues from its ``last.ckpt`` when it was stopped. The
files of ``init_run`` are only read. This uses the image and the volume of ``modal_app.py``
under the app name ``chatchtg-object-finetune``, so it does not mix with that app's runs.
"""

from __future__ import annotations

import os

import modal

from object_detector.modal_app import MODEL_CLASSES, VOLUME_DIR, image, volume

APP_NAME = "chatchtg-object-finetune"
FINETUNE_GPU = os.environ.get("OBJECT_DETECTOR_FINETUNE_GPU", "L40S")

app = modal.App(APP_NAME)


@app.function(
    image=image,
    volumes={VOLUME_DIR: volume},
    gpu=FINETUNE_GPU,
    cpu=8,
    memory=49152,
    timeout=24 * 60 * 60,
    retries=modal.Retries(max_retries=5, initial_delay=0.0),
)
def finetune(
    run_name: str,
    init_run: str = "large-1024",
    size: str = "large",
    resolution: int = 1536,
    epochs: int = 12,
    batch_size: int = 2,
    grad_accum_steps: int = 8,
    lr: float = 5e-5,
    dataset: str = "cubicasa-v1",
    limit: int = 0,
) -> None:
    """Train ``runs/<run_name>`` from the best weights of ``runs/<init_run>``."""
    import json
    import shutil
    import time
    from pathlib import Path

    import rfdetr

    from object_detector import dataset as datasets

    if resolution % 64:
        raise ValueError("The resolution must be a multiple of 64 (square_resize_div_64).")
    run_dir = Path(f"{VOLUME_DIR}/runs/{run_name}")
    if run_name == init_run:
        raise ValueError("The new run must have its own name.")
    dataset_dir = Path(f"{VOLUME_DIR}/datasets/{dataset}")
    if limit:
        datasets.subset_dataset(dataset_dir, Path("/tmp/subset"), limit)
        dataset_dir = Path("/tmp/subset")
    run_dir.mkdir(parents=True, exist_ok=True)
    # A copy, so that nothing of the run that we start from is opened for writing.
    start_weights = Path("/tmp/init.pth")
    shutil.copyfile(f"{VOLUME_DIR}/runs/{init_run}/checkpoint_best_total.pth", start_weights)
    settings = {
        "init_run": init_run,
        "size": size,
        "resolution": resolution,
        "epochs": epochs,
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum_steps,
        "lr": lr,
        "dataset": dataset,
        "limit": limit,
        "gpu": FINETUNE_GPU,
    }
    (run_dir / "run_settings.json").write_text(json.dumps(settings, indent=2))
    checkpoint = run_dir / "last.ckpt"
    print("resuming from", checkpoint if checkpoint.exists() else start_weights, flush=True)

    start = time.time()
    model = getattr(rfdetr, MODEL_CLASSES[size])(pretrain_weights=str(start_weights), resolution=resolution)
    model.train(
        dataset_dir=str(dataset_dir),
        output_dir=str(run_dir),
        epochs=epochs,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        resolution=resolution,
        lr=lr,
        lr_encoder=lr * 1.5,
        resume=str(checkpoint) if checkpoint.exists() else None,
        early_stopping=True,
        early_stopping_patience=5,
        tensorboard=True,
    )
    minutes = (time.time() - start) / 60
    print(f"training took {minutes:.1f} minutes", flush=True)
    (run_dir / "train_minutes.txt").write_text(f"{minutes:.1f}\n")
    volume.commit()
