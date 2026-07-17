from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ._cache import cache_dir as _cache_dir

DEFAULT_HF_REPO_ID = "meituan-longcat/LongCat-Audio-Codec"

LongCatDecoderName = Literal[
    "16k_4codebooks",
    "24k_2codebooks",
    "24k_4codebooks",
]

CHECKPOINT_FILENAMES = {
    "encoder": "LongCatAudioCodec_encoder.pt",
    "encoder_cmvn": "LongCatAudioCodec_encoder_cmvn.npy",
    "decoder_16k_4codebooks": "LongCatAudioCodec_decoder_16k_4codebooks.pt",
    "decoder_24k_2codebooks": "LongCatAudioCodec_decoder_24k_2codebooks.pt",
    "decoder_24k_4codebooks": "LongCatAudioCodec_decoder_24k_4codebooks.pt",
}

CONFIG_STEMS = {
    "encoder": "LongCatAudioCodec_encoder",
    "decoder_16k_4codebooks": "LongCatAudioCodec_decoder_16k_4codebooks",
    "decoder_24k_2codebooks": "LongCatAudioCodec_decoder_24k_2codebooks",
    "decoder_24k_4codebooks": "LongCatAudioCodec_decoder_24k_4codebooks",
}

DECODER_CONFIG_KEYS: dict[LongCatDecoderName, str] = {
    "16k_4codebooks": "decoder_16k_4codebooks",
    "24k_2codebooks": "decoder_24k_2codebooks",
    "24k_4codebooks": "decoder_24k_4codebooks",
}


@dataclass(frozen=True)
class LongCatConfigPaths:
    encoder: Path
    decoder_16k_4codebooks: Path
    decoder_24k_2codebooks: Path
    decoder_24k_4codebooks: Path

    def decoder(self, name: LongCatDecoderName) -> Path:
        if name == "16k_4codebooks":
            return self.decoder_16k_4codebooks
        if name == "24k_2codebooks":
            return self.decoder_24k_2codebooks
        if name == "24k_4codebooks":
            return self.decoder_24k_4codebooks
        raise ValueError(f"Unknown LongCat decoder: {name}.")


@dataclass(frozen=True)
class LongCatAssets:
    cache_dir: Path
    ckpt_dir: Path
    configs: LongCatConfigPaths
    checkpoints: Mapping[str, Path]


def ensure_longcat_assets(
    cache_dir: str | os.PathLike[str] | None = None,
    *,
    decoders: Sequence[LongCatDecoderName] = tuple(DECODER_CONFIG_KEYS),
    repo_id: str = DEFAULT_HF_REPO_ID,
    local_files_only: bool = False,
    force_download: bool = False,
) -> LongCatAssets:
    decoder_names = _validate_decoders(decoders)
    root = _cache_dir(cache_dir)
    ckpt_dir = root / "ckpts"
    config_dir = root / "configs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_keys = _required_checkpoint_keys(decoder_names)
    checkpoints = {
        name: _ensure_checkpoint(
            repo_id=repo_id,
            cache_dir=root,
            ckpt_dir=ckpt_dir,
            filename=CHECKPOINT_FILENAMES[name],
            local_files_only=local_files_only,
            force_download=force_download,
        )
        for name in checkpoint_keys
    }
    configs = write_longcat_configs(config_dir, checkpoints, decoders=decoder_names)

    return LongCatAssets(
        cache_dir=root,
        ckpt_dir=ckpt_dir,
        configs=configs,
        checkpoints=checkpoints,
    )


def write_longcat_configs(
    config_dir: str | os.PathLike[str],
    checkpoints: Mapping[str, Path],
    *,
    decoders: Sequence[LongCatDecoderName] = tuple(DECODER_CONFIG_KEYS),
) -> LongCatConfigPaths:
    config_root = Path(config_dir)
    config_root.mkdir(parents=True, exist_ok=True)
    decoder_names = _validate_decoders(decoders)
    config_keys = ("encoder", *(DECODER_CONFIG_KEYS[name] for name in decoder_names))

    paths = {key: config_root / f"{stem}.yaml" for key, stem in CONFIG_STEMS.items()}
    for key in config_keys:
        stem = CONFIG_STEMS[key]
        data = _read_default_config(stem)
        ckpt_key = key.removeprefix("decoder_")
        checkpoint_key = key if key == "encoder" else f"decoder_{ckpt_key}"
        data["codec_config"]["ckpt_path"] = str(checkpoints[checkpoint_key])
        _write_yaml(paths[key], data)

    return LongCatConfigPaths(
        encoder=paths["encoder"],
        decoder_16k_4codebooks=paths["decoder_16k_4codebooks"],
        decoder_24k_2codebooks=paths["decoder_24k_2codebooks"],
        decoder_24k_4codebooks=paths["decoder_24k_4codebooks"],
    )


def _required_checkpoint_keys(decoders: Sequence[LongCatDecoderName]) -> tuple[str, ...]:
    keys = ["encoder", "encoder_cmvn"]
    keys.extend(DECODER_CONFIG_KEYS[name] for name in decoders)
    return tuple(dict.fromkeys(keys))


def _validate_decoders(decoders: Sequence[LongCatDecoderName]) -> tuple[LongCatDecoderName, ...]:
    if isinstance(decoders, str):
        raise TypeError("decoders must be a sequence of decoder names, not a string.")
    if not decoders:
        raise ValueError("decoders must not be empty.")
    unknown = [name for name in decoders if name not in DECODER_CONFIG_KEYS]
    if unknown:
        raise ValueError(f"Unknown LongCat decoders: {unknown}.")
    return tuple(dict.fromkeys(decoders))


def _ensure_checkpoint(
    *,
    repo_id: str,
    cache_dir: Path,
    ckpt_dir: Path,
    filename: str,
    local_files_only: bool,
    force_download: bool,
) -> Path:
    target = ckpt_dir / filename
    if target.exists() and not force_download:
        return target

    hf_hub_download = _require_huggingface_hub()
    downloaded = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=f"ckpts/{filename}",
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


def _read_default_config(stem: str) -> dict[str, object]:
    try:
        from longcat_audio_codec import default_config_path
    except ImportError as exc:
        raise ImportError(
            "LongCat integration requires the installable fork. Install "
            "`anytrain[longcat]` or install "
            "`longcat-audio-codec @ git+https://github.com/rookiejune/LongCat-Audio-Codec.git`."
        ) from exc

    path = Path(default_config_path(stem))
    yaml = _require_yaml()
    with path.open() as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or "codec_config" not in data:
        raise ValueError(f"Invalid LongCat config: {path}.")
    return data


def _write_yaml(path: Path, data: Mapping[str, object]) -> None:
    yaml = _require_yaml()
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    os.replace(tmp, path)


def _require_huggingface_hub():
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "LongCat checkpoint download requires `huggingface-hub`. Install `anytrain[longcat]`."
        ) from exc
    return hf_hub_download


def _require_yaml():
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "LongCat config writing requires PyYAML. Install `anytrain[longcat]`."
        ) from exc
    return yaml
