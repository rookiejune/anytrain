from __future__ import annotations

import os
from pathlib import Path
from typing import TypedDict

from ._cache import cache_dir as _cache_dir

DEFAULT_HF_REPO_ID = "Yidiii/UniCodec_ckpt"
DEFAULT_CONFIG_NAME = "unicodec_frame75_10s_nq1_code16384_dim512_acousitic.yaml"
DEFAULT_CHECKPOINT_FILENAME = "unicode.ckpt"


class UniCodecAssets(TypedDict):
    cache_dir: Path
    config: Path
    checkpoint: Path


def ensure_unicodec_assets(
    cache_dir: str | os.PathLike[str] | None = None,
    *,
    repo_id: str = DEFAULT_HF_REPO_ID,
    config_name: str = DEFAULT_CONFIG_NAME,
    checkpoint_filename: str = DEFAULT_CHECKPOINT_FILENAME,
    local_files_only: bool = False,
    force_download: bool = False,
) -> UniCodecAssets:
    root = _cache_dir(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    checkpoint = _ensure_checkpoint(
        repo_id=repo_id,
        cache_dir=root,
        filename=checkpoint_filename,
        local_files_only=local_files_only,
        force_download=force_download,
    )

    return {
        "cache_dir": root,
        "config": _default_config_path(config_name),
        "checkpoint": checkpoint,
    }


def _ensure_checkpoint(
    *,
    repo_id: str,
    cache_dir: Path,
    filename: str,
    local_files_only: bool,
    force_download: bool,
) -> Path:
    target = cache_dir / filename
    if target.exists() and not force_download:
        return target

    hf_hub_download = _require_huggingface_hub()
    downloaded = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(cache_dir),
            local_files_only=local_files_only,
            force_download=force_download,
        )
    )
    if not target.exists():
        raise FileNotFoundError(
            f"Expected Hugging Face download to create {target}, but got {downloaded}."
        )
    return target


def _default_config_path(name: str) -> Path:
    try:
        from unicodec import default_config_path
    except ImportError as exc:
        raise ImportError(
            "UniCodec integration requires the installable UniCodec fork. Install "
            "`anytrain[unicodec]` or install "
            "`unicodec @ git+https://github.com/rookiejune/UniCodec.git`."
        ) from exc

    path = Path(default_config_path(name))
    if not path.exists():
        raise FileNotFoundError(f"UniCodec config does not exist: {path}.")
    return path


def _require_huggingface_hub():
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "UniCodec checkpoint download requires `huggingface-hub`. "
            "Install `anytrain[unicodec]`."
        ) from exc
    return hf_hub_download
