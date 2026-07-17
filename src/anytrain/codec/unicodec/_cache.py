from __future__ import annotations

import os
from pathlib import Path

from ...env import DEFAULT_HF_HOME as DEFAULT_HF_HOME
from ...env import HF_HOME_ENV as HF_HOME_ENV
from ...env import hf_home

__all__ = [
    "DEFAULT_HF_HOME",
    "HF_HOME_ENV",
]


def cache_dir(cache_dir: str | os.PathLike[str] | None = None) -> Path:
    if cache_dir is not None:
        return Path(cache_dir).expanduser()

    return hf_home() / "unicodec"
