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
    modal_endpoint: str | None = None
    """The hostname of the Modal endpoint that the modal provider calls."""
    modal_proxy_token: str | None = field(default=None, repr=False)
    """A Modal proxy token in its combined form, ``wk-<id>.ws-<secret>``."""
    judge_model: str | None = None
    """The model of the eval judge. None means the judge's default."""
    room_counter: str = "off"
    """``off``: the LLM answers everything. ``confident``: code answers a room-count question
    when it is confident. ``always``: code answers every room-count question."""
    object_counter: str = "off"
    """``off``: the LLM answers everything. ``detections``: code answers an object-count question
    about a Finnish or a Swiss plan from the detections stored for the drawing, when it has them."""
    object_counter_skip: str | None = None
    """The kinds that the object counter leaves to the LLM, as ``profile:kind`` separated by
    commas (``swiss:door,swiss:window``); empty: none. None means ``counter.DEFAULT_SKIP``."""

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
            modal_endpoint=env.get("CHATCHTG_MODAL_ENDPOINT") or None,
            modal_proxy_token=env.get("CHATCHTG_MODAL_PROXY_TOKEN") or None,
            judge_model=env.get("CHATCHTG_JUDGE_MODEL") or None,
            room_counter=env.get("CHATCHTG_ROOM_COUNTER", "off"),
            object_counter=env.get("CHATCHTG_OBJECT_COUNTER", "off"),
            object_counter_skip=env.get("CHATCHTG_OBJECT_COUNTER_SKIP"),
        )
