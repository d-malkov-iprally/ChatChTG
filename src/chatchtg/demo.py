"""The demo set: hand-picked drawings, each with the questions to ask about it.

The set is the file ``demo_set.json`` next to this module. Each drawing names a file of the
benchmark by its path below the data root. A drawing whose file is not on this machine is
left out, so the app works without the dataset.

The answers to the questions can be saved ahead of a recording (``chatchtg demo-answers``).
The app then shows the saved answer of a chip after a short wait, and asks nothing.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from chatchtg.documents.library import DocumentLibrary, UnsupportedDocumentError
from chatchtg.documents.metadata import DocumentRecord
from chatchtg.schemas import AnswerResponse

logger = logging.getLogger(__name__)

DEMO_SET = Path(__file__).with_name("demo_set.json")
SAVED_ANSWERS_FILE = "demo-answers.json"
"""The file of the saved answers, in the state folder."""
REPLAY_WAIT_SECONDS = 1.0
"""How long the app waits before it shows a saved answer."""


@dataclass(frozen=True)
class DemoQuestion:
    label: str
    """A few words for the chip."""
    text: str
    """The question that is asked."""
    reference: str | None = None
    """The benchmark's answer, with its unit, for the presenter to compare with."""


@dataclass(frozen=True)
class DemoDrawing:
    path: str
    """Below the data root, for example ``floorplans/data/swiss_dwellings/x.png``."""
    title: str
    note: str
    """One line on what this drawing shows well."""
    questions: tuple[DemoQuestion, ...]


def load_demo_set(path: Path | None = None) -> list[DemoDrawing]:
    """The drawings of the demo set, in the order of the file. Without ``path``: ``DEMO_SET``."""
    path = DEMO_SET if path is None else path
    return [
        DemoDrawing(
            path=drawing["path"],
            title=drawing["title"],
            note=drawing.get("note", ""),
            questions=tuple(
                DemoQuestion(
                    label=question["label"], text=question["text"], reference=question.get("reference")
                )
                for question in drawing["questions"]
            ),
        )
        for drawing in json.loads(path.read_text(encoding="utf-8"))
    ]


def register_demo_set(
    library: DocumentLibrary, data_root: Path, drawings: list[DemoDrawing]
) -> list[tuple[DemoDrawing, DocumentRecord]]:
    """Register the drawings whose file is below ``data_root``; leave out the others.

    The record has the attributes that ``chatchtg ingest`` gives the same file.
    """
    root = data_root.resolve()
    registered = []
    for drawing in drawings:
        path = (root / drawing.path).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            continue
        try:
            record = library.register_file(path, attributes=_attributes(drawing.path))
        except UnsupportedDocumentError:
            continue
        registered.append((drawing, record))
    return registered


def _attributes(path: str) -> dict[str, str]:
    """``<dataset>/data/<source>/<file>`` gives the path in the dataset, the source and its licence."""
    parts = Path(path).parts
    if len(parts) < 3 or parts[1] != "data":
        return {}
    attributes = {"original_path": Path(*parts[1:]).as_posix()}
    if len(parts) > 3:
        attributes["source"] = parts[2]
        attributes["licence_file"] = f"data/{parts[2]}/LICENSE.md"
    return attributes


class SavedAnswers:
    """The saved answers of the demo questions: one JSON file.

    ``made_with`` names the settings that decide an answer. The file records them, and a file
    that was made with other settings gives no answers, so the app never shows an answer that
    the current settings would not give.
    """

    def __init__(self, path: Path, made_with: dict[str, str | None]) -> None:
        self.path = path
        self._made_with = made_with

    def load(self) -> dict[tuple[str, str], AnswerResponse]:
        """The answers by (document ID, question). Empty when there is no usable file."""
        if not self.path.is_file():
            return {}
        try:
            saved = json.loads(self.path.read_text(encoding="utf-8"))
            if saved["made_with"] != self._made_with:
                logger.warning(
                    "The saved demo answers in %s were made with other settings (%s), so they "
                    "are not used. Run `chatchtg demo-answers` again.",
                    self.path,
                    saved["made_with"],
                )
                return {}
            return {
                (item["document_id"], item["message"]): AnswerResponse.model_validate(item["response"])
                for item in saved["answers"]
            }
        except (ValueError, KeyError, TypeError) as error:
            logger.warning("The saved demo answers in %s cannot be read: %s", self.path, error)
            return {}

    def save(self, answers: dict[tuple[str, str], AnswerResponse]) -> None:
        """Replace the file with these answers."""
        saved = {
            "made_with": self._made_with,
            "answers": [
                {"document_id": document_id, "message": message, "response": response.model_dump(mode="json")}
                for (document_id, message), response in answers.items()
            ],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        partial = self.path.with_name(self.path.name + ".partial")
        partial.write_text(json.dumps(saved, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(partial, self.path)  # a reader never sees half a file
