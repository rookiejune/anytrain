from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from .._audio import resample
from .._module import DeviceModule
from .assets import BiCodecAssets, ensure_bicodec_assets

SAMPLE_RATE = 16_000
NUM_CHANNELS = 1
FEATURE_HIDDEN_STATE_INDEXES = (11, 14, 16)
BICODEC_INSTALL_HINT = (
    'Clone https://github.com/SparkAudio/Spark-TTS, add it to PYTHONPATH, and install '
    'wrapper dependencies with `python -m pip install "einx>=0.3" "huggingface-hub>=0.23" '
    '"omegaconf>=2.3" "safetensors>=0.5" "soundfile>=0.12" "soxr>=0.5" '
    '"torchaudio>=2.5" "transformers>=4.46"`.'
)


@dataclass(frozen=True)
class BiCodecTokens:
    semantic: Tensor
    global_tokens: Tensor


class BiCodec(DeviceModule):
    num_channels: int = NUM_CHANNELS

    def __init__(
        self,
        model: nn.Module,
        processor: Any,
        feature_extractor: nn.Module,
        device: torch.device,
        *,
        model_dir: Path | None = None,
        assets: BiCodecAssets | None = None,
        config: dict[str, Any] | None = None,
        feature_hidden_state_indexes: tuple[int, ...] = FEATURE_HIDDEN_STATE_INDEXES,
    ) -> None:
        super().__init__()

        if not feature_hidden_state_indexes:
            raise ValueError("feature_hidden_state_indexes must not be empty.")

        self.model = model
        self.processor = processor
        self.feature_extractor = feature_extractor
        self.model_dir = model_dir
        self.assets = assets
        self.config = {} if config is None else config
        self.feature_hidden_state_indexes = feature_hidden_state_indexes
        self.sample_rate = int(self.config.get("sample_rate", SAMPLE_RATE))
        self.ref_segment_length = _ref_segment_length(self.config)
        self.semantic_codebook_sizes = _codebook_sizes(getattr(model, "quantizer", None))
        self.global_codebook_sizes = _codebook_sizes(getattr(model, "speaker_encoder", None))

        extractor_config = getattr(self.feature_extractor, "config", None)
        if extractor_config is not None:
            extractor_config.output_hidden_states = True
        self._init_device(device)

    @classmethod
    def from_pretrained(
        cls,
        *,
        cache_dir: str | os.PathLike[str] | None = None,
        model_dir: str | os.PathLike[str] | None = None,
        repo_id: str = "SparkAudio/Spark-TTS-0.5B",
        device: str | torch.device | None = None,
        local_files_only: bool = False,
        force_download: bool = False,
    ) -> BiCodec:
        assets: BiCodecAssets | None = None
        if model_dir is None:
            assets = ensure_bicodec_assets(
                cache_dir=cache_dir,
                repo_id=repo_id,
                local_files_only=local_files_only,
                force_download=force_download,
            )
            root = assets["model_dir"]
        else:
            root = Path(model_dir).expanduser()
            if not root.exists():
                raise FileNotFoundError(f"BiCodec model directory does not exist: {root}.")

        resolved_device = _device(device)
        model_cls = _load_bicodec_model()
        processor_cls, feature_extractor_cls = _load_wav2vec2_classes()
        config = _load_config(root / "config.yaml")

        model = model_cls.load_from_checkpoint(root / "BiCodec").to(resolved_device)
        model.eval()
        processor = processor_cls.from_pretrained(str(root / "wav2vec2-large-xlsr-53"))
        feature_extractor = feature_extractor_cls.from_pretrained(
            str(root / "wav2vec2-large-xlsr-53")
        ).to(resolved_device)
        feature_extractor.eval()

        return cls(
            model=model,
            processor=processor,
            feature_extractor=feature_extractor,
            device=resolved_device,
            model_dir=root,
            assets=assets,
            config=config,
        )

    @torch.no_grad()
    def encode(
        self,
        audio: Tensor,
        sample_rate: int,
        *,
        ref_audio: Tensor | None = None,
        ref_sample_rate: int | None = None,
    ) -> BiCodecTokens:
        return self.tokenize(
            audio,
            sample_rate,
            ref_audio=ref_audio,
            ref_sample_rate=ref_sample_rate,
        )

    @torch.no_grad()
    def tokenize(
        self,
        audio: Tensor,
        sample_rate: int,
        *,
        ref_audio: Tensor | None = None,
        ref_sample_rate: int | None = None,
    ) -> BiCodecTokens:
        wav = self._audio(audio, sample_rate)
        ref_wav = (
            self._audio(ref_audio, ref_sample_rate or sample_rate)
            if ref_audio is not None
            else wav
        )
        ref_wav = self._reference(ref_wav)
        features = self.extract_features(wav)

        semantic, global_tokens = self.model.tokenize(
            {
                "wav": wav.to(self.device),
                "ref_wav": ref_wav.to(self.device),
                "feat": features.to(self.device),
            }
        )
        tokens = BiCodecTokens(
            semantic=_integer_tensor(semantic, "semantic tokens"),
            global_tokens=_integer_tensor(global_tokens, "global tokens"),
        )
        self._validate_tokens(tokens)
        return tokens

    @torch.no_grad()
    def extract_features(self, wav: Tensor) -> Tensor:
        if wav.dim() != 2:
            raise ValueError("wav must have shape [batch, time].")

        waveforms = [item.detach().cpu().numpy() for item in wav]
        inputs = self.processor(
            waveforms,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
            output_hidden_states=True,
        ).input_values
        output = self.feature_extractor(inputs.to(self.device))
        hidden_states = output.hidden_states
        if hidden_states is None:
            raise ValueError("BiCodec wav2vec2 feature extractor did not return hidden states.")
        indexes = self.feature_hidden_state_indexes
        if len(hidden_states) <= max(indexes):
            raise ValueError(
                "BiCodec wav2vec2 feature extractor returned too few hidden states: "
                f"{len(hidden_states)}."
            )

        features = sum(hidden_states[index] for index in indexes) / len(indexes)
        if not isinstance(features, Tensor):
            raise TypeError("BiCodec wav2vec2 mixed features must be a Tensor.")
        if features.dim() != 3:
            raise ValueError("BiCodec wav2vec2 mixed features must have shape [batch, time, dim].")
        return features

    @torch.no_grad()
    def decode(self, tokens: BiCodecTokens) -> Tensor:
        return self.detokenize(tokens.semantic, tokens.global_tokens)

    @torch.no_grad()
    def detokenize(self, semantic: Tensor, global_tokens: Tensor) -> Tensor:
        tokens = BiCodecTokens(
            semantic=_integer_tensor(semantic, "semantic tokens"),
            global_tokens=_integer_tensor(global_tokens, "global tokens"),
        )
        self._validate_tokens(tokens)
        audio = self.model.detokenize(
            tokens.semantic.to(self.device),
            tokens.global_tokens.to(self.device),
        )
        if not isinstance(audio, Tensor):
            raise TypeError("BiCodec detokenize must return a Tensor.")
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        if audio.dim() != 3:
            raise ValueError("BiCodec detokenize must return audio shape [batch, channels, time].")
        return audio

    @torch.no_grad()
    def reconstruct(
        self,
        audio: Tensor,
        sample_rate: int,
        *,
        ref_audio: Tensor | None = None,
        ref_sample_rate: int | None = None,
    ) -> Tensor:
        return self.decode(
            self.encode(
                audio,
                sample_rate,
                ref_audio=ref_audio,
                ref_sample_rate=ref_sample_rate,
            )
        )

    def _audio(self, audio: Tensor, sample_rate: int) -> Tensor:
        if audio.dim() != 3:
            raise ValueError("BiCodec encode expects audio shape [batch, channels, time].")
        if audio.size(1) != NUM_CHANNELS:
            raise ValueError("BiCodec expects mono audio with shape [batch, 1, time].")
        if audio.size(-1) <= 0:
            raise ValueError("BiCodec audio must contain at least one sample.")
        return resample(audio, sample_rate, self.sample_rate).squeeze(1)

    def _reference(self, wav: Tensor) -> Tensor:
        length = self.ref_segment_length
        if length is None:
            return wav
        if wav.size(-1) <= 0:
            raise ValueError("BiCodec reference audio must contain at least one sample.")
        if wav.size(-1) < length:
            repeats = math.ceil(length / wav.size(-1))
            wav = wav.repeat(1, repeats)
        return wav[:, :length]

    def _validate_tokens(self, tokens: BiCodecTokens) -> None:
        if tokens.semantic.dim() < 2:
            raise ValueError("semantic tokens must include batch and time dimensions.")
        if tokens.global_tokens.dim() < 2:
            raise ValueError("global tokens must include a batch dimension.")
        if tokens.semantic.size(0) != tokens.global_tokens.size(0):
            raise ValueError("semantic and global tokens must align on batch.")


def _ref_segment_length(config: dict[str, Any]) -> int | None:
    duration = config.get("ref_segment_duration")
    hop_length = config.get("latent_hop_length")
    sample_rate = int(config.get("sample_rate", SAMPLE_RATE))
    if duration is None or hop_length is None:
        return None
    length = int(sample_rate * float(duration))
    hop = int(hop_length)
    if hop <= 0:
        raise ValueError("latent_hop_length must be positive.")
    return max(hop, length // hop * hop)


def _integer_tensor(value: Any, name: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a Tensor.")
    if value.dtype == torch.bool or torch.is_floating_point(value) or torch.is_complex(value):
        raise TypeError(f"{name} must contain integer ids.")
    return value


def _codebook_sizes(module: Any) -> tuple[int, ...]:
    if module is None:
        return ()
    value = getattr(module, "codebook_sizes", None)
    if value is not None:
        return tuple(int(size) for size in value)
    value = getattr(module, "codebook_size", None)
    if value is not None:
        count = int(getattr(module, "num_codebooks", 1))
        return (int(value),) * count
    for name in ("num_embeddings", "n_tokens", "num_tokens", "vocab_size"):
        value = getattr(module, name, None)
        if value is not None:
            return (int(value),)
    return ()


def _device(device: str | torch.device | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_bicodec_model() -> Any:
    try:
        from sparktts.models.bicodec import BiCodec as UpstreamBiCodec
    except ImportError as exc:
        raise ImportError(
            "BiCodec requires the Spark-TTS source package on PYTHONPATH and its "
            f"runtime dependencies. {BICODEC_INSTALL_HINT}"
        ) from exc
    return UpstreamBiCodec


def _load_config(path: Path) -> dict[str, Any]:
    try:
        from sparktts.utils.file import load_config
    except ImportError as exc:
        raise ImportError(
            "BiCodec config loading requires the Spark-TTS source package on PYTHONPATH."
        ) from exc
    config = load_config(str(path))
    try:
        from omegaconf import DictConfig, OmegaConf
    except ImportError:
        DictConfig = None
        OmegaConf = None
    if DictConfig is not None and isinstance(config, DictConfig):
        config = OmegaConf.to_container(config, resolve=True)
    if not isinstance(config, dict):
        raise TypeError(f"BiCodec config must be a dict: {path}.")
    return config


def _load_wav2vec2_classes() -> tuple[Any, Any]:
    try:
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
    except ImportError as exc:
        raise ImportError(
            'BiCodec feature extraction requires transformers. Install it with `python -m pip install "transformers>=4.46"`.'
        ) from exc
    return Wav2Vec2FeatureExtractor, Wav2Vec2Model


__all__ = [
    "BiCodec",
    "BiCodecTokens",
    "FEATURE_HIDDEN_STATE_INDEXES",
    "NUM_CHANNELS",
    "SAMPLE_RATE",
]
