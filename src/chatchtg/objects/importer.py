"""Store the detections of a batch prediction as extractions, one file per drawing.

Usage: python -m chatchtg.objects.importer <detections.jsonl> [<images_dir>] [--version N]

``detections.jsonl`` is the output of ``object_detector/modal_predict.py``. ``images_dir`` is
the folder of the benchmark's CubiCasa5K images (default
``$CHATCHTG_DATA_ROOT/floorplans/data/cubicasa5k``). A plan whose image is there is stored
under the hash of that file, and only if the hash is the one that the prediction read; the
other plans are stored under the hash of the image that the prediction read. The extractions
go to the extraction store of the settings (``var/metadata``), as ``--version`` of the
producer (default: the version that the app reads). The run name and the checkpoint are read
from ``settings.json`` next to the detections.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from chatchtg.config import Settings
from chatchtg.extractions.models import Extraction, ImageSize, Item
from chatchtg.extractions.store import ExtractionStore
from chatchtg.objects.counter import PRODUCER, PRODUCER_VERSION

SOURCE_FOLDER = "data/cubicasa5k"
"""The folder of the CubiCasa5K plans in the paths of the benchmark's questions."""


def to_extraction(
    row: dict[str, Any], params: dict[str, Any] | None = None, version: int = PRODUCER_VERSION
) -> Extraction:
    """One line of ``detections.jsonl`` as an extraction."""
    items = []
    per_label: Counter[str] = Counter()
    ordered = sorted(row["detections"], key=lambda d: (d["label"], -d["score"]))
    for detection in ordered:
        label = detection["label"]
        per_label[label] += 1
        items.append(
            Item(
                id=f"{label.replace(' ', '-')}-{per_label[label]}",
                kind="object",
                label=label,
                box=tuple(detection["box"]),
                score=min(max(detection["score"], 0.0), 1.0),
            )
        )
    return Extraction(
        document_id=row["sha256"],
        producer=PRODUCER,
        version=version,
        image=ImageSize(width=row["width"], height=row["height"]),
        params=params or {},
        source=f"{SOURCE_FOLDER}/{row['image']}",
        attributes={
            "cubicasa_split": row["split"],
            "cubicasa_plan": row["plan"],
            # Some images are in CubiCasa5K twice, under two plans, sometimes of two splits.
            "cubicasa_splits": {row["image"]: row["split"]},
            # The detector was trained on the plans of the train split that are in its dataset.
            "in_training_data": _trained_on(row),
        },
        items=items,
    )


def _trained_on(row: dict[str, Any]) -> bool:
    return row["split"] == "train" and row.get("in_dataset", True)


def merge_duplicate(extraction: Extraction, row: dict[str, Any]) -> Extraction:
    """Add a second plan with the same image to its extraction. The detections are the same."""
    attributes = dict(extraction.attributes)
    attributes["cubicasa_splits"] = {**attributes["cubicasa_splits"], row["image"]: row["split"]}
    attributes["in_training_data"] = attributes["in_training_data"] or _trained_on(row)
    return extraction.model_copy(update={"attributes": attributes})


def import_detections(
    detections: Path,
    images_dir: Path | None,
    store: ExtractionStore,
    params: dict[str, Any],
    version: int = PRODUCER_VERSION,
) -> Counter[str]:
    """Store every plan of ``detections``. Returns how many plans went which way."""
    outcome: Counter[str] = Counter()
    extractions: dict[str, Extraction] = {}
    with detections.open(encoding="utf-8") as lines:
        for line in lines:
            row = json.loads(line)
            local = images_dir / row["image"] if images_dir else None
            if local is not None and local.is_file():
                if hashlib.sha256(local.read_bytes()).hexdigest() != row["sha256"]:
                    # The boxes would not fit this image.
                    print(f"Not stored, the image differs: {local}", file=sys.stderr)
                    outcome["image differs"] += 1
                    continue
                outcome["in the benchmark"] += 1
            else:
                outcome["not in the benchmark"] += 1
            known = extractions.get(row["sha256"])
            if known is None:
                extractions[row["sha256"]] = to_extraction(row, params, version)
            else:
                outcome["same image as another plan"] += 1
                extractions[row["sha256"]] = merge_duplicate(known, row)
    for extraction in extractions.values():
        store.put(extraction)
    return outcome


def main(argv: list[str]) -> int:
    from chatchtg.wiring import build_extraction_store

    settings = Settings.from_env()
    version = PRODUCER_VERSION
    if "--version" in argv:
        at = argv.index("--version")
        version = int(argv[at + 1])
        argv = argv[:at] + argv[at + 2 :]
    detections = Path(argv[0])
    if len(argv) > 1:
        images_dir = Path(argv[1])
    else:
        images_dir = settings.data_root / "floorplans" / "data" / "cubicasa5k"
    params: dict[str, Any] = {"run": "large-1024", "checkpoint": "checkpoint_best_total.pth"}
    settings_file = detections.with_name("settings.json")
    if settings_file.is_file():
        params.update(json.loads(settings_file.read_text(encoding="utf-8")))
    outcome = import_detections(
        detections, images_dir if images_dir.is_dir() else None, build_extraction_store(settings), params, version
    )
    print(f"{PRODUCER}@{version}:", ", ".join(f"{name}: {n}" for name, n in outcome.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
