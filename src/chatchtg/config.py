"""Runtime settings, read from environment variables."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    data_root: Path
    """Root of the downloaded datasets. Referenced documents are stored relative to it."""
    state_dir: Path
    """Where uploaded blobs, the metadata database and eval results are written."""
    composer: str
    llm_provider: str
    llm_model: str | None = None
    """The provider's model. None means the adapter's default."""

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        env = os.environ if env is None else env
        return cls(
            data_root=Path(env.get("CHATCHTG_DATA_ROOT", "data")).resolve(),
            state_dir=Path(env.get("CHATCHTG_STATE_DIR", "var")).resolve(),
            composer=env.get("CHATCHTG_COMPOSER", "baseline"),
            llm_provider=env.get("CHATCHTG_LLM_PROVIDER", "fake"),
            llm_model=env.get("CHATCHTG_LLM_MODEL") or None,
        )
