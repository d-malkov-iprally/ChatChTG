"""Extraction storage: one JSON file per drawing and producer version.

``<root>/<document_id>/<producer>@<version>.json``
"""

from __future__ import annotations

import logging
import re
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

from pydantic import ValidationError

from chatchtg.extractions.models import Extraction, Item, ItemRef

logger = logging.getLogger(__name__)
_DOCUMENT_ID = re.compile(r"[0-9a-f]{64}")
_PRODUCER = re.compile(r"[a-z0-9][a-z0-9-]*")

_FILE = re.compile(r"(?P<producer>[a-z0-9][a-z0-9-]*)@(?P<version>[1-9][0-9]*)\.json")


class ExtractionStore(Protocol):
    def put(self, extraction: Extraction) -> None:
        """Store the extraction. It replaces one of the same document, producer and version."""

    def get(
        self, document_id: str, producer: str, version: int | None = None
    ) -> Extraction | None:
        """Return the extraction, or None if the producer did not run on this document.

        Without ``version``, return the highest stored version.
        """

    def list(self, document_id: str) -> list[Extraction]:
        """All extractions of a document, ordered by producer and version."""

    def iter(self, producer: str, version: int) -> Iterator[Extraction]:
        """The extractions of one producer version, for each document that has one."""

    def resolve(self, document_id: str, ref: ItemRef | str) -> Item | None:
        """Return the item that the reference points at, or None if it is not stored."""


class LocalExtractionStore:
    def __init__(self, root: Path) -> None:
        self._root = root

    def put(self, extraction: Extraction) -> None:
        target = self._path(extraction.document_id, extraction.producer, extraction.version)
        target.parent.mkdir(parents=True, exist_ok=True)
        # A name of its own per writer: a reader must never see a file that is half written.
        tmp = target.with_name(f"{target.name}.{uuid.uuid4().hex}.tmp")
        tmp.write_text(extraction.model_dump_json(by_alias=True, indent=1), encoding="utf-8")
        tmp.replace(target)

    def get(
        self, document_id: str, producer: str, version: int | None = None
    ) -> Extraction | None:
        if version is None:
            versions = [v for p, v in self._stored(document_id) if p == producer]
            if not versions:
                return None
            version = max(versions)
        return self._load(self._path(document_id, producer, version))

    def list(self, document_id: str) -> list[Extraction]:
        loaded = (
            self._load_or_skip(self._path(document_id, producer, version))
            for producer, version in sorted(self._stored(document_id))
        )
        return [extraction for extraction in loaded if extraction is not None]

    def iter(self, producer: str, version: int) -> Iterator[Extraction]:
        if not self._root.is_dir():
            return
        for folder in sorted(self._root.iterdir()):
            extraction = self._load_or_skip(folder / f"{producer}@{version}.json")
            if extraction is not None:
                yield extraction

    def resolve(self, document_id: str, ref: ItemRef | str) -> Item | None:
        if isinstance(ref, str):
            ref = ItemRef.parse(ref)
        extraction = self.get(document_id, ref.producer, ref.version)
        return extraction.item(ref.item_id) if extraction else None

    def _path(self, document_id: str, producer: str, version: int) -> Path:
        # The names become a path. Anything else than a hash and a producer name, such as
        # "../x", would read a file outside the store.
        if _DOCUMENT_ID.fullmatch(document_id) is None:
            raise ValueError(f"Not a document ID: {document_id!r}")
        if _PRODUCER.fullmatch(producer) is None:
            raise ValueError(f"Not a producer name: {producer!r}")
        return self._root / document_id / f"{producer}@{int(version)}.json"

    def _stored(self, document_id: str) -> list[tuple[str, int]]:
        """The (producer, version) pairs that have a file for this document."""
        if _DOCUMENT_ID.fullmatch(document_id) is None:
            raise ValueError(f"Not a document ID: {document_id!r}")
        folder = self._root / document_id
        if not folder.is_dir():
            return []
        matches = (_FILE.fullmatch(path.name) for path in folder.iterdir())
        return [(m["producer"], int(m["version"])) for m in matches if m]

    def _load_or_skip(self, path: Path) -> Extraction | None:
        """Load a file for a listing. One file that cannot be read must not hide the others."""
        try:
            return self._load(path)
        except ValidationError as error:
            logger.warning("Skipped the extraction file %s: %s", path, error)
            return None

    @staticmethod
    def _load(path: Path) -> Extraction | None:
        try:
            text = path.read_text(encoding="utf-8")
        except (FileNotFoundError, NotADirectoryError):
            return None
        return Extraction.model_validate_json(text)
