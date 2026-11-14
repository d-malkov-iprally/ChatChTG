"""Open-vocabulary object detection: find the objects that a text prompt names."""

from chatchtg.detection.detector import Detection, Detector, count_by_label, parse_labels
from chatchtg.detection.drawing import draw_detections
from chatchtg.detection.models import MODELS, ModelSpec, load_detector

__all__ = [
    "MODELS",
    "Detection",
    "Detector",
    "ModelSpec",
    "count_by_label",
    "draw_detections",
    "load_detector",
    "parse_labels",
]
