"""Runtime settings, read from environment variables."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import dotenv_values

ENV_FILE = ".env"


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
    anthropic_api_key: str | None = field(default=None, repr=False)
    """This project's Anthropic key. None means the Anthropic SDK's own credential lookup."""
    judge_model: str | None = None
    """The model of the eval judge. None means the judge's default."""

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        """Read the settings from ``env``.

        Without ``env``, read the ``.env`` file of the current directory and the process
        environment. A variable set in both takes its value from the process environment.
        """
        if env is None:
            file_values = {k: v for k, v in dotenv_values(ENV_FILE).items() if v is not None}
            env = {**file_values, **os.environ}
        return cls(
            data_root=Path(env.get("CHATCHTG_DATA_ROOT", "data")).resolve(),
            state_dir=Path(env.get("CHATCHTG_STATE_DIR", "var")).resolve(),
            composer=env.get("CHATCHTG_COMPOSER", "baseline"),
            llm_provider=env.get("CHATCHTG_LLM_PROVIDER", "fake"),
            llm_model=env.get("CHATCHTG_LLM_MODEL") or None,
            anthropic_api_key=env.get("CHATCHTG_ANTHROPIC_API_KEY") or None,
            judge_model=env.get("CHATCHTG_JUDGE_MODEL") or None,
        )
