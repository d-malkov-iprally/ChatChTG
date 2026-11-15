"""The demo set: hand-picked drawings, each with the questions to ask about it.

The set is the file ``demo_set.json`` next to this module. Each drawing names a file of the
benchmark by its path below the data root. A drawing whose file is not on this machine is
left out, so the app works without the dataset.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from chatchtg.documents.library import DocumentLibrary, UnsupportedDocumentError
from chatchtg.documents.metadata import DocumentRecord

DEMO_SET = Path(__file__).with_name("demo_set.json")


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
