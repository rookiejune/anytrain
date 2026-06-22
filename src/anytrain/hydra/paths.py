from __future__ import annotations

from pathlib import Path
from typing import Any


def get_field(cfg: Any, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    value = cfg.get(key, default) if hasattr(cfg, "get") else getattr(cfg, key, default)
    return default if value is None else value


def logger_fields(cfg: Any) -> dict[str, Any]:
    return {
        "save_dir": get_field(cfg, "save_dir", "outputs"),
        "name": get_field(cfg, "name", "anytrain"),
        "version": get_field(cfg, "version", "default"),
    }


def logger_root(*, save_dir: str | Path, name: str, version: str | int | None) -> str:
    root = Path(str(save_dir)) / str(name)
    return str(root if version is None else root / str(version))
