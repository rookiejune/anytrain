from __future__ import annotations

import os
from pathlib import Path

HF_HOME_ENV = "HF_HOME"
DEFAULT_HF_HOME = Path.home() / ".cache" / "huggingface"


def resolve_longcat_cache_dir(cache_dir: str | os.PathLike[str] | None = None) -> Path:
    if cache_dir is not None:
        return Path(cache_dir).expanduser()

    return _hf_home() / "longcat-audio-codec"


def _hf_home() -> Path:
    value = os.environ.get(HF_HOME_ENV)
    if value is None:
        return DEFAULT_HF_HOME
    if value == "":
        raise ValueError(f"{HF_HOME_ENV} must not be empty.")
    return Path(value).expanduser()
