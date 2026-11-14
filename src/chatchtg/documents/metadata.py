"""Metadata storage: one record per document, with free-form attributes."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field


class DocumentRecord(BaseModel):
    id: str
    filename: str
    media_type: str
    size_bytes: int
    sha256: str
    location: str
    attributes: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class MetadataStore(Protocol):
    def upsert(self, record: DocumentRecord) -> None: ...

    def get(self, document_id: str) -> DocumentRecord | None: ...

    def list(self, limit: int = 100, offset: int = 0) -> list[DocumentRecord]:
        """Newest first."""

    def count(self) -> int: ...

    def patch_attributes(self, document_id: str, patch: dict[str, Any]) -> DocumentRecord | None:
        """Merge ``patch`` into the attributes. A ``None`` value removes the key."""


_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id          TEXT PRIMARY KEY,
    filename    TEXT NOT NULL,
    media_type  TEXT NOT NULL,
    size_bytes  INTEGER NOT NULL,
    sha256      TEXT NOT NULL,
    location    TEXT NOT NULL,
    attributes  TEXT NOT NULL,
    created_at  TEXT NOT NULL
)
"""
_COLUMNS = "id, filename, media_type, size_bytes, sha256, location, attributes, created_at"


class SqliteMetadataStore:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn, conn:
            conn.execute(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        # One connection per operation: safe across Streamlit's threads.
        return sqlite3.connect(self._db_path)

    def upsert(self, record: DocumentRecord) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute(
                f"INSERT OR REPLACE INTO documents ({_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record.id,
                    record.filename,
                    record.media_type,
                    record.size_bytes,
                    record.sha256,
                    record.location,
                    json.dumps(record.attributes),
                    record.created_at.isoformat(),
                ),
            )

    def get(self, document_id: str) -> DocumentRecord | None:
        with closing(self._connect()) as conn:
            row = conn.execute(
                f"SELECT {_COLUMNS} FROM documents WHERE id = ?", (document_id,)
            ).fetchone()
        return _to_record(row) if row else None

    def list(self, limit: int = 100, offset: int = 0) -> list[DocumentRecord]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                f"SELECT {_COLUMNS} FROM documents ORDER BY created_at DESC, id LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [_to_record(row) for row in rows]

    def count(self) -> int:
        with closing(self._connect()) as conn:
            return conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]

    def patch_attributes(self, document_id: str, patch: dict[str, Any]) -> DocumentRecord | None:
        with closing(self._connect()) as conn, conn:
            # Take the write lock before reading, so a concurrent patch is not lost.
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT attributes FROM documents WHERE id = ?", (document_id,)
            ).fetchone()
            if row is None:
                return None
            attributes = json.loads(row[0])
            for key, value in patch.items():
                if value is None:
                    attributes.pop(key, None)
                else:
                    attributes[key] = value
            conn.execute(
                "UPDATE documents SET attributes = ? WHERE id = ?",
                (json.dumps(attributes), document_id),
            )
        return self.get(document_id)


def _to_record(row: tuple) -> DocumentRecord:
    return DocumentRecord(
        id=row[0],
        filename=row[1],
        media_type=row[2],
        size_bytes=row[3],
        sha256=row[4],
        location=row[5],
        attributes=json.loads(row[6]),
        created_at=datetime.fromisoformat(row[7]),
    )
