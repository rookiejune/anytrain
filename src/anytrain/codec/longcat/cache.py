from __future__ import annotations

import os
from pathlib import Path

ANYTRAIN_LONGCAT_CACHE_ENV = "ANYTRAIN_LONGCAT_CACHE"
ANYTRAIN_CACHE_ENV = "ANYTRAIN_CACHE_DIR"


def resolve_longcat_cache_dir(cache_dir: str | os.PathLike[str] | None = None) -> Path:
    if cache_dir is not None:
        return Path(cache_dir).expanduser()

    longcat_cache = _env_path(ANYTRAIN_LONGCAT_CACHE_ENV)
    if longcat_cache is not None:
        return longcat_cache

    anytrain_cache = _env_path(ANYTRAIN_CACHE_ENV)
    if anytrain_cache is not None:
        return anytrain_cache / "longcat-audio-codec"

    return Path.home() / ".cache" / "anytrain" / "longcat-audio-codec"


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    if value is None:
        return None
    if value == "":
        raise ValueError(f"{name} must not be empty.")
    return Path(value).expanduser()

