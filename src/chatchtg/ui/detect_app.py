"""Streamlit demo of open-vocabulary detection: select an image, name the objects, see the boxes."""

from __future__ import annotations

import os
import time
from pathlib import Path

import streamlit as st
from PIL import Image

from chatchtg.config import Settings
from chatchtg.detection import (
    UnreadableImageError,
    count_by_label,
    draw_detections,
    models,
    open_image,
    parse_labels,
)
from chatchtg.documents.library import RASTER_SUFFIXES

MAX_LISTED = 2000
# SAM 3 runs the model once for each label, so the number of labels limits the work.
MAX_LABELS = 20

st.set_page_config(page_title="Detection demo", layout="wide")


@st.cache_resource(max_entries=2, show_spinner=False)
def detector(model_name: str):
    return models.load_detector(model_name)


def inside(path: Path, root: Path) -> bool:
    """True if ``path``, with its symbolic links followed, is ``root`` or is below it."""
    try:
        return path.resolve().is_relative_to(root.resolve())
    except (OSError, ValueError):  # for example a NUL character in the name
        return False


@st.cache_data(ttl=60, show_spinner=False)
def list_images(folder: str) -> list[str]:
    """The first ``MAX_LISTED`` PNG/JPEG files below ``folder``, as relative paths.

    The walk is in sorted order and stops when it has ``MAX_LISTED`` files.
    """
    root = Path(folder)
    names: list[str] = []
    for directory, subdirectories, files in os.walk(root):
        subdirectories.sort()
        for file in sorted(files):
            if Path(file).suffix.lower() in RASTER_SUFFIXES:
                names.append((Path(directory) / file).relative_to(root).as_posix())
                if len(names) == MAX_LISTED:
                    return names
    return names


st.title("Open-vocabulary detection demo")
st.caption("Select an image, name the objects to find, and compare what each detector finds.")

image: Image.Image | None = None
image_id = ""
source = st.segmented_control(
    "Image", ["Upload", "Dataset folder"], default="Upload", required=True, key="source"
)
if source == "Upload":
    upload = st.file_uploader("Image (PNG or JPEG)", type=["png", "jpg", "jpeg"], key="upload")
    if upload is not None:
        try:
            image = open_image(upload)
            image_id = f"upload:{upload.file_id}"
        except UnreadableImageError as error:
            st.error(str(error))
else:
    # The page reads files only below the data folder, so that a visitor of the page
    # cannot list or open other files of this machine.
    data_root = Settings.from_env().data_root
    subfolder = st.text_input(
        "Subfolder of the data folder",
        value="",
        placeholder="for example floorplans/data/cubicasa5k",
        help=f"The data folder is {data_root}. Leave this empty to list all of it.",
        key="folder",
    )
    folder = data_root / subfolder.strip()
    if not inside(folder, data_root):
        st.error("The folder must be inside the data folder.")
    elif not folder.is_dir():
        st.info("This folder does not exist.")
    elif names := list_images(str(folder)):
        name = st.selectbox(f"Image (first {MAX_LISTED:,} of the folder)", names, key="image_name")
        if inside(folder / name, data_root):
            try:
                image = open_image(folder / name)
                image_id = f"file:{folder}/{name}"
            except UnreadableImageError as error:
                st.error(str(error))
        else:
            st.error("This file is a link to a file outside the data folder.")
    else:
        st.info("No PNG or JPEG files in this folder.")

model_name = st.segmented_control(
    "Model", list(models.MODELS), default=next(iter(models.MODELS)), required=True, key="model"
)
spec = models.MODELS[model_name]
if spec.classes:
    st.caption(
        f"{model_name} is not open-vocabulary. It finds only: {', '.join(spec.classes)}. "
        "It was trained on the CubiCasa5K floor plans, so its results on those plans are "
        "better than on other plans."
    )

with st.form("detect"):
    prompt = st.text_input(
        "Objects to detect",
        value="window, door",
        help=f"Separate the objects with commas, at most {MAX_LABELS}. Short English nouns work best.",
        max_chars=500,
        key="prompt",
    )
    threshold = st.slider(
        "Minimum score",
        min_value=0.02,
        max_value=0.9,
        value=spec.default_threshold,
        step=0.01,
        help="Scores of different models are not comparable. OWLv2 scores are much lower.",
        # The key holds the family, so that a change of family resets the slider to its default.
        key=f"threshold-{spec.family}",
    )
    submitted = st.form_submit_button("Detect", type="primary", disabled=image is None)

if image is not None:
    labels = parse_labels(prompt)
    if submitted and not labels:
        st.warning("Name at least one object to detect.")
    elif submitted and len(labels) > MAX_LABELS:
        st.warning(f"Name at most {MAX_LABELS} objects. The prompt has {len(labels)}.")
    elif submitted:
        try:
            with st.spinner(f"Loading {model_name}. The first load downloads the weights."):
                loaded = detector(model_name)
        except models.DetectorUnavailableError as error:
            st.error(str(error))
            st.stop()
        with st.spinner("Detecting..."):
            start = time.perf_counter()
            try:
                detections = loaded.detect(image, labels, threshold)
            except Exception as error:  # for example an image too small for the model
                st.error(f"{model_name} failed on this image: {error}")
                st.stop()
            seconds = time.perf_counter() - start
        unknown = models.unknown_labels(spec, labels)
        st.session_state["result"] = {
            "unknown": unknown,
            "image_id": image_id,
            "model": model_name,
            "labels": labels,
            "threshold": threshold,
            "detections": detections,
            "seconds": seconds,
        }

    result = st.session_state.get("result")
    if result is not None and result["image_id"] != image_id:
        # The result belongs to another image.
        result = None

    image_column, result_column = st.columns([2, 1])
    with image_column:
        if result is None:
            st.image(image, width="stretch", alt="The selected image")
        else:
            st.image(
                draw_detections(image, result["detections"]),
                width="stretch",
                alt="The selected image with a box around each detection",
            )
        st.caption(f"{image.width} x {image.height} pixels")
    with result_column:
        if result is None:
            st.info("Select **Detect** to run the model on this image.")
        else:
            detections = result["detections"]
            st.metric("Detections", len(detections))
            st.caption(
                f"{result['model']} · minimum score {result['threshold']:.2f} · "
                f"{result['seconds']:.1f} s · prompt: {', '.join(result['labels'])}"
            )
            if result["unknown"]:
                st.warning(f"{result['model']} does not know: {', '.join(result['unknown'])}.")
            if detections:
                counts = count_by_label(detections)
                st.dataframe(
                    {"label": list(counts), "count": list(counts.values())},
                    hide_index=True,
                    alt="Number of detections of each label",
                )
                st.dataframe(
                    {
                        "label": [detection.label for detection in detections],
                        "score": [round(detection.score, 3) for detection in detections],
                        "left": [round(detection.box[0]) for detection in detections],
                        "top": [round(detection.box[1]) for detection in detections],
                        "right": [round(detection.box[2]) for detection in detections],
                        "bottom": [round(detection.box[3]) for detection in detections],
                    },
                    hide_index=True,
                    alt="All detections, with score and box in pixels",
                )
                if models.MODELS[result["model"]].family == "grounding-dino":
                    st.caption(
                        "A label of two or more prompt words, such as \"window door\", means that "
                        "Grounding DINO matched the box to each of them."
                    )
            else:
                st.info("Nothing found. Try a lower minimum score or other words.")
