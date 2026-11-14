"""The floor-plan model of CubiCasa5K as a detector.

The model is not open-vocabulary: it finds the classes of ``classes.CLASSES`` only. It was
trained on the CubiCasa5K floor plans, so its results on those plans do not show how well it
works on other plans. Code and weights are CC BY-NC 4.0, see LICENSE in this folder.
"""

from __future__ import annotations

import hashlib
import ssl
import urllib.request
from pathlib import Path

import certifi
from PIL import Image

from chatchtg.detection.cubicasa.classes import ICON_CLASSES, ROOM_CLASSES, canonical_label
from chatchtg.detection.detector import Detection

WEIGHTS_URL = (
    "https://drive.usercontent.google.com/download"
    "?id=1gRB7ez1e4H7a9Y09lLqRuna0luZO5VRK&export=download&confirm=t"
)
WEIGHTS_SHA256 = "dd20b4e1bf1d670f2125107b079df06958b1ccd36e49a464ab739aeb00b8e7a2"
WEIGHTS_FILE = "cubicasa5k-model_best_val_loss_var.pkl"

# The channels of the model's output: 21 heatmaps, 12 room classes, 11 icon classes.
SPLIT = [21, 12, 11]
# The threshold of the corner heatmaps, as in the samples of CubiCasa5K.
HEATMAP_THRESHOLD = 0.2
# A larger image is scaled down to this longer side, to limit memory and time.
MAX_SIDE = 2048
# The model reads the image as it is and turned by 90, 180 and 270 degrees: (turn, turn back).
ROTATIONS = [(0, 0), (1, -1), (2, 2), (-1, 1)]


class WeightsError(RuntimeError):
    """The weights could not be downloaded, or the file is not the expected file."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_weights(folder: Path) -> Path:
    """Return the weights file in ``folder``. Download it (209 MB) if it is not there."""
    path = folder / WEIGHTS_FILE
    if not path.is_file():
        folder.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".part")
        # The Python of python.org on macOS has no root certificates of its own.
        certificates = ssl.create_default_context(cafile=certifi.where())
        try:
            with (
                urllib.request.urlopen(WEIGHTS_URL, timeout=60, context=certificates) as response,
                partial.open("wb") as file,
            ):
                while chunk := response.read(1 << 20):
                    file.write(chunk)
        except OSError as error:
            partial.unlink(missing_ok=True)
            raise WeightsError(f"The CubiCasa5K weights could not be downloaded: {error}") from error
        if _sha256(partial) != WEIGHTS_SHA256:
            partial.unlink()
            raise WeightsError(
                "The downloaded CubiCasa5K weights are not the expected file. Download "
                "model_best_val_loss_var.pkl from the README of "
                f"https://github.com/CubiCasa/CubiCasa5k and save it as {path}."
            )
        partial.rename(path)
    elif _sha256(path) != WEIGHTS_SHA256:
        raise WeightsError(f"{path} is not the expected CubiCasa5K weights file. Delete it to download it again.")
    return path


class CubiCasaDetector:
    def __init__(self, weights: Path, device: str) -> None:
        import torch

        from chatchtg.detection.cubicasa.model import hg_furukawa_original

        # The file is a pickle. weights_only refuses a pickle that would run code.
        checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
        model = hg_furukawa_original(n_classes=51)
        model.conv4_ = torch.nn.Conv2d(256, sum(SPLIT), bias=True, kernel_size=1)
        model.upsample = torch.nn.ConvTranspose2d(sum(SPLIT), sum(SPLIT), kernel_size=4, stride=4)
        model.load_state_dict(checkpoint["model_state"])
        self.model = model.to(device).eval()
        self.device = device

    def detect(self, image: Image.Image, labels: list[str], threshold: float) -> list[Detection]:
        wanted = {name for name in map(canonical_label, labels) if name}
        if not wanted:
            return []
        image = image.convert("RGB")
        scale = min(1.0, MAX_SIDE / max(image.size))
        if scale < 1:
            image = image.resize((round(image.width * scale), round(image.height * scale)))
        detections = [
            Detection(label, score, tuple(value / scale for value in box))
            for label, score, box in self._find(image)
            if label in wanted and score >= threshold
        ]
        return sorted(detections, key=lambda detection: -detection.score)

    def _find(self, image: Image.Image) -> list[tuple[str, float, tuple[float, float, float, float]]]:
        """All icons, openings and rooms of ``image`` as (class name, score, box)."""
        import numpy as np
        import torch
        import torch.nn.functional as F
        from skimage import draw

        from chatchtg.detection.cubicasa.postprocessing import get_polygons, split_prediction
        from chatchtg.detection.cubicasa.rotation import RotateNTurns

        pixels = torch.tensor(np.moveaxis(np.asarray(image, dtype=np.float32), -1, 0))
        pixels = (2 * (pixels / 255.0) - 1).unsqueeze(0).to(self.device)
        height, width = pixels.shape[2:]
        rotate = RotateNTurns()
        predictions = []
        with torch.no_grad():
            for turn, turn_back in ROTATIONS:
                prediction = self.model(rotate(pixels, "tensor", turn))
                prediction = rotate(prediction, "tensor", turn_back)
                prediction = rotate(prediction, "points", turn_back)
                prediction = F.interpolate(
                    prediction, size=(height, width), mode="bilinear", align_corners=True
                )
                predictions.append(prediction[0].cpu())
        mean = torch.mean(torch.stack(predictions), 0, keepdim=True)
        heatmaps, rooms, icons = split_prediction(mean, (height, width), SPLIT)
        room_probabilities = rooms.copy()  # get_polygons changes ``rooms``
        polygons, types, room_polygons, room_types = get_polygons(
            (heatmaps, rooms, icons), HEATMAP_THRESHOLD, [1, 2]
        )

        found = []
        for polygon, kind in zip(polygons, types):
            # Windows and doors are icons too: get_polygons gives them the icon classes 1 and 2.
            name = ICON_CLASSES[int(kind["class"])] if kind["type"] == "icon" else None
            if name:
                xs, ys = polygon[:, 0], polygon[:, 1]
                box = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
                # "prob" is the class evidence divided by the area. It can be a little above 1.
                found.append((name, min(1.0, float(kind["prob"])), box))
        for polygon, kind in zip(room_polygons, room_types):
            name = ROOM_CLASSES[int(kind["class"])]
            if not name:
                continue
            # Rooms of one class come as one shape. Each separate part of it is one room.
            for part in getattr(polygon, "geoms", [polygon]):
                if part.is_empty or part.geom_type != "Polygon":
                    continue
                xs, ys = part.exterior.coords.xy
                rows, columns = draw.polygon(np.array(ys), np.array(xs), shape=(height, width))
                if len(rows) == 0:
                    continue
                # The score of a room is the mean probability of its class inside the room.
                score = float(room_probabilities[int(kind["class"])][rows, columns].mean())
                left, top, right, bottom = part.bounds
                found.append((name, score, (left, top, right, bottom)))
        return found
