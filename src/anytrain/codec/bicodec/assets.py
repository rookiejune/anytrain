from __future__ import annotations

import os
from pathlib import Path
from typing import TypedDict

from ._cache import cache_dir as _cache_dir

DEFAULT_HF_REPO_ID = "SparkAudio/Spark-TTS-0.5B"
SNAPSHOT_PATTERNS = (
    "config.yaml",
    "BiCodec/*",
    "wav2vec2-large-xlsr-53/*",
)


class BiCodecAssets(TypedDict):
    cache_dir: Path
    model_dir: Path


def ensure_bicodec_assets(
    cache_dir: str | os.PathLike[str] | None = None,
    *,
    repo_id: str = DEFAULT_HF_REPO_ID,
    local_files_only: bool = False,
    force_download: bool = False,
) -> BiCodecAssets:
    root = _cache_dir(cache_dir)
    root.mkdir(parents=True, exist_ok=True)

    snapshot_download = _require_huggingface_hub()
    model_dir = Path(
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(root),
            allow_patterns=SNAPSHOT_PATTERNS,
            local_files_only=local_files_only,
            force_download=force_download,
        )
    )
    _validate_model_dir(model_dir)
    return {
        "cache_dir": root,
        "model_dir": model_dir,
    }


def _validate_model_dir(model_dir: Path) -> None:
    required = (
        model_dir / "config.yaml",
        model_dir / "BiCodec" / "config.yaml",
        model_dir / "BiCodec" / "model.safetensors",
        model_dir / "wav2vec2-large-xlsr-53" / "config.json",
    )
    missing = [path for path in required if not path.exists()]
    if missing:
        formatted = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"BiCodec model directory is incomplete: {formatted}.")


def _require_huggingface_hub():
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "BiCodec checkpoint download requires huggingface-hub. "
            'Install it with `python -m pip install "huggingface-hub>=0.23"`.'
        ) from exc
    return snapshot_download
