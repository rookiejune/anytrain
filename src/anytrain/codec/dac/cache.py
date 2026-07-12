from __future__ import annotations

import os
from pathlib import Path

from ...env import ANYTRAIN_HOME_ENV as ANYTRAIN_HOME_ENV
from ...env import DEFAULT_ANYTRAIN_HOME as DEFAULT_ANYTRAIN_HOME
from ...env import anytrain_home

DEFAULT_DAC_HOME = DEFAULT_ANYTRAIN_HOME / "dac"


def resolve_dac_cache_dir(cache_dir: str | os.PathLike[str] | None = None) -> Path:
    if cache_dir is not None:
        return Path(cache_dir).expanduser()
    return anytrain_home() / "dac"


__all__ = [
    "ANYTRAIN_HOME_ENV",
    "DEFAULT_DAC_HOME",
    "resolve_dac_cache_dir",
]
