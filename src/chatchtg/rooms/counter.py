"""The room counter that the answer service uses."""

from __future__ import annotations

from pathlib import Path

from chatchtg.rooms import ocr, router
from chatchtg.rooms.router import Finding


class LabelRoomCounter:
    """Counts rooms from the printed labels and the shapes of the drawing.

    Reading labels needs Apple Vision. Where it is missing, or the drawing cannot be read,
    ``solve`` returns None and the LLM answers.
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        self._cache_dir = cache_dir

    def solve(self, content: bytes, message: str) -> Finding | None:
        try:
            return router.solve(content, message, self._cache_dir)
        except (ocr.OcrUnavailableError, OSError, ValueError):
            return None
