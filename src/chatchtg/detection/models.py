"""The open-vocabulary detectors of the demo. They run locally with Hugging Face transformers.

torch and transformers are in the optional ``detect`` extra. They are imported when a
detector is loaded, so that the rest of the package works without them.
"""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from chatchtg.config import Settings
from chatchtg.detection.cubicasa.classes import CLASSES, canonical_label
from chatchtg.detection.detector import Detection, Detector


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    """"grounding-dino", "owlv2", "sam3" or "cubicasa": it selects how the model is run."""
    checkpoint: str
    """The model's id on the Hugging Face hub, or the address of its project."""
    default_threshold: float
    """A starting threshold. OWLv2 gives much lower scores than Grounding DINO."""
    gated: bool = False
    """True if the weights need an accepted licence and a Hugging Face login."""
    classes: tuple[str, ...] = ()
    """The only classes that the model finds. Empty for an open-vocabulary model."""


MODELS: dict[str, ModelSpec] = {
    spec.name: spec
    for spec in [
        ModelSpec("Grounding DINO tiny", "grounding-dino", "IDEA-Research/grounding-dino-tiny", 0.25),
        ModelSpec("Grounding DINO base", "grounding-dino", "IDEA-Research/grounding-dino-base", 0.25),
        ModelSpec(
            "MM Grounding DINO base",
            "grounding-dino",
            "openmmlab-community/mm_grounding_dino_base_all",
            0.25,
        ),
        ModelSpec("OWLv2 base", "owlv2", "google/owlv2-base-patch16-ensemble", 0.1),
        ModelSpec("OWLv2 large", "owlv2", "google/owlv2-large-patch14-ensemble", 0.1),
        # SAM 3.1 (March 2026) changes the video tracker and has no transformers weights.
        # For one image and a text prompt, SAM 3 is the newest SAM that transformers loads.
        ModelSpec("SAM 3", "sam3", "facebook/sam3", 0.3, gated=True),
        # Not open-vocabulary: a floor-plan model with fixed classes. It is here to compare.
        ModelSpec("CubiCasa5K", "cubicasa", "https://github.com/CubiCasa/CubiCasa5k", 0.2, classes=CLASSES),
    ]
}


class DetectorUnavailableError(RuntimeError):
    """The detector cannot be loaded. The message tells the user what to do."""


class MissingDependencyError(DetectorUnavailableError):
    """torch or transformers is not installed."""


class ModelAccessError(DetectorUnavailableError):
    """The weights are gated and this machine has no access to them."""


def default_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_detector(name: str, device: str | None = None) -> Detector:
    """Load the model ``name`` of ``MODELS``. The first load downloads its weights."""
    spec = MODELS[name]
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as error:
        raise MissingDependencyError(
            "The detectors need torch and transformers. Install them with `uv sync --extra detect`."
        ) from error
    if spec.family == "cubicasa":
        return _load_cubicasa(device or default_device())
    try:
        return HuggingFaceDetector(spec, device or default_device())
    except OSError as error:
        if not spec.gated:
            raise
        # transformers reports a gated or unreachable repository as an OSError.
        raise ModelAccessError(
            f"{spec.name} could not be loaded. Its weights are gated: request access at "
            f"https://huggingface.co/{spec.checkpoint}, wait for the approval, then log in "
            f"with `uv run --extra detect hf auth login`. The error was: {error}"
        ) from error


def _load_cubicasa(device: str) -> Detector:
    try:
        from chatchtg.detection import cubicasa

        # Import now what the detector imports at its first detection.
        from chatchtg.detection.cubicasa import postprocessing  # noqa: F401
    except ImportError as error:
        raise MissingDependencyError(
            "The CubiCasa5K model needs scikit-image and shapely. "
            "Install them with `uv sync --extra detect`."
        ) from error
    try:
        weights = cubicasa.ensure_weights(Settings.from_env().state_dir / "models")
    except cubicasa.WeightsError as error:
        raise DetectorUnavailableError(str(error)) from error
    return cubicasa.CubiCasaDetector(weights, device)


def unknown_labels(spec: ModelSpec, labels: list[str]) -> list[str]:
    """The labels that a model with fixed classes does not know. Empty for the other models."""
    if not spec.classes:
        return []
    return [label for label in labels if canonical_label(label) is None]


class HuggingFaceDetector:
    def __init__(self, spec: ModelSpec, device: str) -> None:
        from transformers import AutoModel, AutoModelForZeroShotObjectDetection, AutoProcessor

        self.spec = spec
        self.device = device
        self.processor = AutoProcessor.from_pretrained(spec.checkpoint)
        model_class = AutoModel if spec.family == "sam3" else AutoModelForZeroShotObjectDetection
        self.model = model_class.from_pretrained(spec.checkpoint)
        self.model.to(device).eval()

    def detect(self, image: Image.Image, labels: list[str], threshold: float) -> list[Detection]:
        if not labels:
            return []
        image = image.convert("RGB")
        if self.spec.family == "owlv2":
            found = self._detect_owlv2(image, labels, threshold)
        elif self.spec.family == "sam3":
            found = self._detect_sam3(image, labels, threshold)
        else:
            found = self._detect_grounding_dino(image, labels, threshold)
        width, height = image.size
        detections = []
        for label, score, (left, top, right, bottom) in found:
            if not label:
                # Grounding DINO found a box but no prompt word passed the text threshold.
                continue
            box = (max(0.0, left), max(0.0, top), min(float(width), right), min(float(height), bottom))
            detections.append(Detection(label=label, score=score, box=box))
        return sorted(detections, key=lambda detection: -detection.score)

    def _detect_grounding_dino(self, image: Image.Image, labels: list[str], threshold: float):
        import torch

        # Grounding DINO reads one lower-case string, with a full stop after each label.
        text = " ".join(f"{label} ." for label in labels)
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        result = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=threshold,
            text_threshold=threshold,
            target_sizes=[image.size[::-1]],
        )[0]
        return _rows(result)

    def _detect_owlv2(self, image: Image.Image, labels: list[str], threshold: float):
        import torch

        inputs = self.processor(text=[labels], images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # OWLv2 pads the image to a square at the bottom and the right before it resizes it.
        # Its boxes are relative to that square, not to the image.
        side = max(image.size)
        result = self.processor.post_process_grounded_object_detection(
            outputs, threshold=threshold, target_sizes=[(side, side)], text_labels=[labels]
        )[0]
        return _rows(result)


    def _detect_sam3(self, image: Image.Image, labels: list[str], threshold: float):
        import torch

        # SAM 3 takes one noun phrase for each run. The image is encoded once and the
        # result is used for each label. Only the boxes are used, not the masks.
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
        found = []
        with torch.no_grad():
            vision = self.model.get_vision_features(pixel_values=pixel_values.to(self.device))
            for label in labels:
                text = self.processor(text=label, return_tensors="pt").to(self.device)
                outputs = self.model(vision_embeds=vision, **text)
                result = self.processor.post_process_object_detection(
                    outputs, threshold=threshold, target_sizes=[image.size[::-1]]
                )[0]
                found += [
                    (label, float(score), [float(value) for value in box])
                    for score, box in zip(result["scores"], result["boxes"])
                ]
        return found


def _rows(result: dict) -> list[tuple[str, float, list[float]]]:
    return [
        (label, float(score), [float(value) for value in box])
        for label, score, box in zip(result["text_labels"], result["scores"], result["boxes"])
    ]
