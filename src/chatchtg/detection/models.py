"""The open-vocabulary detectors of the demo. They run locally with Hugging Face transformers.

torch and transformers are in the optional ``detect`` extra. They are imported when a
detector is loaded, so that the rest of the package works without them.
"""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from chatchtg.detection.detector import Detection, Detector


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    """"grounding-dino" or "owlv2": it selects the prompt format and the post-processing."""
    checkpoint: str
    """The model's id on the Hugging Face hub."""
    default_threshold: float
    """A starting threshold. OWLv2 gives much lower scores than Grounding DINO."""


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
    ]
}


class MissingDependencyError(RuntimeError):
    """torch or transformers is not installed."""


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
    return HuggingFaceDetector(spec, device or default_device())


class HuggingFaceDetector:
    def __init__(self, spec: ModelSpec, device: str) -> None:
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.spec = spec
        self.device = device
        self.processor = AutoProcessor.from_pretrained(spec.checkpoint)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(spec.checkpoint)
        self.model.to(device).eval()

    def detect(self, image: Image.Image, labels: list[str], threshold: float) -> list[Detection]:
        if not labels:
            return []
        image = image.convert("RGB")
        if self.spec.family == "owlv2":
            found = self._detect_owlv2(image, labels, threshold)
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


def _rows(result: dict) -> list[tuple[str, float, list[float]]]:
    return [
        (label, float(score), [float(value) for value in box])
        for label, score, box in zip(result["text_labels"], result["scores"], result["boxes"])
    ]
