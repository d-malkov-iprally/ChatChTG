"""The document library: the only entry point to documents for other modules.

It joins document storage (bytes) and metadata storage (records), and decides
which file types are accepted.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from chatchtg.documents.blobs import BlobStore
from chatchtg.documents.metadata import DocumentRecord, MetadataStore

# Accepted file types and the bytes each one starts with.
# To accept a new format, add it here.
SUPPORTED_MEDIA_TYPES: dict[str, bytes] = {
    "image/png": b"\x89PNG\r\n\x1a\n",
    "image/jpeg": b"\xff\xd8\xff",
}
RASTER_SUFFIXES = frozenset({".png", ".jpg", ".jpeg"})


class UnsupportedDocumentError(ValueError):
    """The file is not one of the accepted types."""


class DocumentNotFoundError(LookupError):
    """No document has this ID."""


def sniff_media_type(data: bytes) -> str:
    for media_type, signature in SUPPORTED_MEDIA_TYPES.items():
        if data.startswith(signature):
            return media_type
    raise UnsupportedDocumentError(
        "Unsupported file type. Accepted types: " + ", ".join(SUPPORTED_MEDIA_TYPES)
    )


class DocumentLibrary:
    def __init__(self, blobs: BlobStore, metadata: MetadataStore) -> None:
        self._blobs = blobs
        self._metadata = metadata

    def add_upload(
        self, data: bytes, filename: str, attributes: dict[str, Any] | None = None
    ) -> DocumentRecord:
        """Store a copy of ``data``. The same content always gets the same ID."""
        return self._add(data, filename, attributes, lambda: self._blobs.put(data))

    def register_file(self, path: Path, attributes: dict[str, Any] | None = None) -> DocumentRecord:
        """Register a file that stays where it is. Registering it again changes nothing."""
        data = path.read_bytes()
        return self._add(data, path.name, attributes, lambda: self._blobs.reference(path))

    def get(self, document_id: str) -> DocumentRecord:
        record = self._metadata.get(document_id)
        if record is None:
            raise DocumentNotFoundError(document_id)
        return record

    def read(self, document_id: str) -> bytes:
        return self._blobs.read(self.get(document_id).location)

    def list(self, limit: int = 100, offset: int = 0) -> list[DocumentRecord]:
        return self._metadata.list(limit=limit, offset=offset)

    def count(self) -> int:
        return self._metadata.count()

    def patch_attributes(self, document_id: str, patch: dict[str, Any]) -> DocumentRecord:
        """Merge ``patch`` into the attributes. A ``None`` value removes the key."""
        record = self._metadata.patch_attributes(document_id, patch)
        if record is None:
            raise DocumentNotFoundError(document_id)
        return record

    def _add(self, data, filename, attributes, store_blob) -> DocumentRecord:
        media_type = sniff_media_type(data)
        digest = hashlib.sha256(data).hexdigest()
        existing = self._metadata.get(digest)
        if existing is not None:
            return existing
        record = DocumentRecord(
            id=digest,
            filename=filename,
            media_type=media_type,
            size_bytes=len(data),
            sha256=digest,
            location=store_blob(),
            attributes=attributes or {},
            created_at=datetime.now(timezone.utc),
        )
        self._metadata.upsert(record)
        return record
