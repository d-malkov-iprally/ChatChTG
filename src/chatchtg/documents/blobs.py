"""Document storage: the bytes of each document.

A location is an opaque string. Only this module knows how to read one.
- ``cas:<sha256>``: a copy owned by the store (uploads).
- ``ref:<path>``: a file that stays where it is (dataset files). The path is
  relative to the data root when the file is inside it.
"""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path
from typing import Protocol


class BlobStore(Protocol):
    def put(self, data: bytes) -> str:
        """Copy ``data`` into the store and return its location."""

    def reference(self, path: Path) -> str:
        """Return a location for an existing file, without copying it."""

    def read(self, location: str) -> bytes: ...

    def exists(self, location: str) -> bool:
        """Whether ``read`` would find the bytes of this location."""


class LocalBlobStore:
    def __init__(self, blob_dir: Path, data_root: Path) -> None:
        self._blob_dir = blob_dir
        self._data_root = data_root.resolve()

    def put(self, data: bytes) -> str:
        digest = hashlib.sha256(data).hexdigest()
        target = self._blob_dir / digest
        if not target.exists():
            self._blob_dir.mkdir(parents=True, exist_ok=True)
            # A name of its own per writer: two uploads of the same content must not share it.
            tmp = target.with_name(f"{digest}.{uuid.uuid4().hex}.tmp")
            tmp.write_bytes(data)
            tmp.replace(target)
        return f"cas:{digest}"

    def reference(self, path: Path) -> str:
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.is_relative_to(self._data_root):
            return f"ref:{path.relative_to(self._data_root).as_posix()}"
        return f"ref:{path.as_posix()}"

    def read(self, location: str) -> bytes:
        return self._path(location).read_bytes()

    def exists(self, location: str) -> bool:
        return self._path(location).is_file()

    def _path(self, location: str) -> Path:
        kind, _, value = location.partition(":")
        if kind == "cas":
            return self._blob_dir / value
        if kind == "ref":
            # Joining with an absolute path yields that absolute path.
            return self._data_root / value
        raise ValueError(f"Unknown blob location: {location!r}")
