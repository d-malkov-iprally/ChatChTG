"""Document storage: the bytes of each document.

A location is an opaque string. Only this module knows how to read one.
- ``cas:<sha256>``: a copy owned by the store (uploads).
- ``ref:<path>``: a file that stays where it is (dataset files). The path is
  relative to the data root when the file is inside it.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol


class BlobStore(Protocol):
    def put(self, data: bytes) -> str:
        """Copy ``data`` into the store and return its location."""

    def reference(self, path: Path) -> str:
        """Return a location for an existing file, without copying it."""

    def read(self, location: str) -> bytes: ...


class LocalBlobStore:
    def __init__(self, blob_dir: Path, data_root: Path) -> None:
        self._blob_dir = blob_dir
        self._data_root = data_root.resolve()

    def put(self, data: bytes) -> str:
        digest = hashlib.sha256(data).hexdigest()
        target = self._blob_dir / digest
        if not target.exists():
            self._blob_dir.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(".tmp")
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
        kind, _, value = location.partition(":")
        if kind == "cas":
            return (self._blob_dir / value).read_bytes()
        if kind == "ref":
            # Joining with an absolute path yields that absolute path.
            return (self._data_root / value).read_bytes()
        raise ValueError(f"Unknown blob location: {location!r}")
