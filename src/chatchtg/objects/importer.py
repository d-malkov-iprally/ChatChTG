"""Store the detections of a batch prediction as extractions, one file per drawing.

Usage:
    python -m chatchtg.objects.importer <detections.jsonl> [<images_dir>] [--version N]
    python -m chatchtg.objects.importer --profile swiss <detections_all.json> [<images_dir>] [--version N]

CubiCasa5K (the default profile): ``detections.jsonl`` is the output of
``object_detector/modal_predict.py``. ``images_dir`` is the folder of the benchmark's CubiCasa5K
images (default ``$CHATCHTG_DATA_ROOT/floorplans/data/cubicasa5k``). A plan whose image is there
is stored under the hash of that file, and only if the hash is the one that the prediction
read; the other plans are stored under the hash of the image that the prediction read. The run
name and the checkpoint are read from ``settings.json`` next to the detections.

Swiss Dwellings (``--profile swiss``): ``detections_all.json`` is the output of
``object_detector/modal_swiss.py::evaluate``. It has no image hashes, so each plan is stored
under the hash of its PNG in ``images_dir`` (default
``$CHATCHTG_DATA_ROOT/floorplans/data/swiss_dwellings``); a plan without its PNG there is not
stored. The file is refused when its minimum scores are not those of the version.

The extractions go to the extraction store of the settings (``var/metadata``), as ``--version``
of the profile's producer (default: the version that the app reads).
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from chatchtg.config import Settings
from PIL import Image

from chatchtg.extractions.models import Extraction, ImageSize, Item
from chatchtg.extractions.store import ExtractionStore
from chatchtg.objects.counter import (
    PRODUCER,
    PRODUCER_VERSION,
    PROFILES,
    RUN_BY_VERSION,
    SWISS,
    DetectorProfile,
)

SOURCE_FOLDER = "data/cubicasa5k"
"""The folder of the CubiCasa5K plans in the paths of the benchmark's questions."""


def to_extraction(
    row: dict[str, Any], params: dict[str, Any] | None = None, version: int = PRODUCER_VERSION
) -> Extraction:
    """One line of ``detections.jsonl`` as an extraction."""
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
        items=to_items(row["detections"]),
    )


def to_items(detections: list[dict[str, Any]]) -> list[Item]:
    """Detections ``{"label", "score", "box"}`` as items, with ids ``<label>-<n>`` by score."""
    items = []
    per_label: Counter[str] = Counter()
    for detection in sorted(detections, key=lambda d: (d["label"], -d["score"])):
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
    return items


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


def swiss_params(data: dict[str, Any], version: int, profile: DetectorProfile = SWISS) -> dict[str, Any]:
    """The parameters stored with each Swiss extraction of ``version``.

    The detections file of the Swiss detector names no run, but it holds the minimum scores
    that were chosen for it on the valid plans. They must be the minimums of ``version``:
    otherwise the counter would count these detections with the minimums of another run.
    """
    expected = profile.thresholds_by_version.get(version)
    if expected is None:
        raise ValueError(f"No minimum scores for {profile.producer}@{version}: add them to its profile")
    found = data.get("thresholds", {})
    differ = {c: found.get(c) for c, t in expected.items() if found.get(c) != t}
    if differ:
        raise ValueError(
            f"The minimum scores of these detections differ from those of {profile.producer}@{version} "
            f"({differ}): they are of another run. Pass the --version of that run."
        )
    return {
        "run_name": profile.runs_by_version[version],
        "checkpoint": "checkpoint_best_total.pth",
        "score_floor": data.get("score_floor"),
        "thresholds": dict(expected),
    }


def import_swiss_detections(
    detections: Path,
    images_dir: Path,
    store: ExtractionStore,
    version: int = SWISS.version,
    profile: DetectorProfile = SWISS,
) -> Counter[str]:
    """Store every plan of a Swiss ``detections_all.json``. Returns how many plans went which way."""
    data = json.loads(detections.read_text(encoding="utf-8"))
    params = swiss_params(data, version, profile)
    outcome: Counter[str] = Counter()
    seen: set[str] = set()
    for plan, row in sorted(data["plans"].items()):
        image = images_dir / f"{plan}.png"
        if not image.is_file():
            print(f"Not stored, no image: {image}", file=sys.stderr)
            outcome["no image"] += 1
            continue
        content = image.read_bytes()
        document_id = hashlib.sha256(content).hexdigest()
        if document_id in seen:
            # Two plans with one image would share one extraction; there are none in the benchmark.
            print(f"Not stored, the same image as another plan: {image}", file=sys.stderr)
            outcome["same image as another plan"] += 1
            continue
        seen.add(document_id)
        with Image.open(image) as opened:
            width, height = opened.size
        store.put(
            Extraction(
                document_id=document_id,
                producer=profile.producer,
                version=version,
                image=ImageSize(width=width, height=height),
                params=params,
                source=f"{profile.source_folder}/{plan}.png",
                attributes={
                    "split": row["split"],
                    # The detector was trained on the plans of the train split.
                    "in_training_data": row["split"] == "train",
                    # From the DXF title, for the reports. The detector does not use it.
                    "dxf_scale": row.get("scale"),
                    "drawing_area": row.get("area"),
                },
                items=to_items(row["detections"]),
            )
        )
        outcome["stored"] += 1
    return outcome


def run_params(settings: dict[str, Any] | None, version: int) -> dict[str, Any]:
    """The parameters stored with each extraction of ``version``.

    ``settings`` is the ``settings.json`` that ``predict_all`` wrote next to the detections,
    or None when there is none. Each version holds the detections of one training run
    (``RUN_BY_VERSION``). Detections of another run are refused: stored under the wrong
    version they would replace the detections that the app reads, with the wrong minimum scores.
    """
    expected = RUN_BY_VERSION.get(version)
    if expected is None:
        raise ValueError(f"No run is named for version {version}: add it to RUN_BY_VERSION")
    params = dict(settings or {})
    run = params.setdefault("run_name", expected)
    if run != expected:
        raise ValueError(
            f"These detections are of the run {run!r}, and version {version} holds the run "
            f"{expected!r}. Pass the --version of that run."
        )
    params.setdefault("checkpoint", "checkpoint_best_total.pth")
    return params


def main(argv: list[str]) -> int:
    from chatchtg.wiring import build_extraction_store

    settings = Settings.from_env()
    name = "cubicasa"
    if "--profile" in argv:
        at = argv.index("--profile")
        name = argv[at + 1]
        argv = argv[:at] + argv[at + 2 :]
    profile = next((p for p in PROFILES if p.name == name), None)
    if profile is None:
        print(f"Unknown profile {name!r}. Available: {', '.join(p.name for p in PROFILES)}", file=sys.stderr)
        return 2
    version = profile.version
    if "--version" in argv:
        at = argv.index("--version")
        version = int(argv[at + 1])
        argv = argv[:at] + argv[at + 2 :]
    if profile is SWISS:
        return _main_swiss(argv, settings, version)
    detections = Path(argv[0])
    if len(argv) > 1:
        images_dir = Path(argv[1])
    else:
        images_dir = settings.data_root / "floorplans" / "data" / "cubicasa5k"
    settings_file = detections.with_name("settings.json")
    try:
        params = run_params(
            json.loads(settings_file.read_text(encoding="utf-8")) if settings_file.is_file() else None,
            version,
        )
    except ValueError as error:
        print(f"Nothing stored. {error}", file=sys.stderr)
        return 2
    outcome = import_detections(
        detections, images_dir if images_dir.is_dir() else None, build_extraction_store(settings), params, version
    )
    print(f"{PRODUCER}@{version}:", ", ".join(f"{name}: {n}" for name, n in outcome.items()))
    return 0


def _main_swiss(argv: list[str], settings: Settings, version: int) -> int:
    from chatchtg.wiring import build_extraction_store

    detections = Path(argv[0])
    images_dir = Path(argv[1]) if len(argv) > 1 else settings.data_root / "floorplans" / "data" / "swiss_dwellings"
    try:
        outcome = import_swiss_detections(detections, images_dir, build_extraction_store(settings), version)
    except ValueError as error:
        print(f"Nothing stored. {error}", file=sys.stderr)
        return 2
    print(f"{SWISS.producer}@{version}:", ", ".join(f"{name}: {n}" for name, n in outcome.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
