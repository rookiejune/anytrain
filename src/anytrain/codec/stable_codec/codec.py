from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any, Literal, Union

import torch
from torch import Tensor, nn
from typing_extensions import TypeAlias

SupportedVersion = Literal["speech-16k", "speech-16k-base"]
PosthocBottleneckPreset = Literal[
    "1x46656_400bps",
    "2x15625_700bps",
    "4x729_1000bps",
]
PosthocBottleneck: TypeAlias = Union[
    PosthocBottleneckPreset,
    Sequence[tuple[Sequence[int], float]],
]

DEFAULT_VERSION: SupportedVersion = "speech-16k"
DEFAULT_PRETRAINED_MODEL = f"stabilityai/stable-codec-{DEFAULT_VERSION}"
SAMPLE_RATE = 16_000
NUM_CHANNELS = 1
DEFAULT_SEMANTIC_VOCAB_SIZE = 46_656


class StableCodec(nn.Module):
    num_channels: int = NUM_CHANNELS
    sample_rate: int = SAMPLE_RATE

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        *,
        posthoc_bottleneck: bool = False,
    ) -> None:
        super().__init__()

        self.model = model
        self.device = device
        self.posthoc_bottleneck = posthoc_bottleneck
        self.sample_rate = int(getattr(model, "sample_rate", SAMPLE_RATE))
        self.semantic_vocab_size = _semantic_vocab_size(model)

    @classmethod
    def from_pretrained(
        cls,
        version: SupportedVersion = DEFAULT_VERSION,
        *,
        pretrained_model: str | None = None,
        device: str | torch.device | None = None,
        posthoc_bottleneck: PosthocBottleneck | None = None,
    ) -> StableCodec:
        model_cls = _load_stable_codec_model()
        resolved_device = _resolve_device(device)
        model = model_cls(
            pretrained_model=pretrained_model or f"stabilityai/stable-codec-{version}",
            device=resolved_device,
        )
        if posthoc_bottleneck is not None:
            model.set_posthoc_bottleneck(posthoc_bottleneck)

        return cls(
            model=model,
            device=resolved_device,
            posthoc_bottleneck=posthoc_bottleneck is not None,
        )

    @classmethod
    def from_config(
        cls,
        model_config_path: str | os.PathLike[str],
        *,
        ckpt_path: str | os.PathLike[str] | None = None,
        device: str | torch.device | None = None,
        posthoc_bottleneck: PosthocBottleneck | None = None,
    ) -> StableCodec:
        model_cls = _load_stable_codec_model()
        resolved_device = _resolve_device(device)
        model = model_cls(
            model_config_path=str(model_config_path),
            ckpt_path=None if ckpt_path is None else str(ckpt_path),
            device=resolved_device,
        )
        if posthoc_bottleneck is not None:
            model.set_posthoc_bottleneck(posthoc_bottleneck)

        return cls(
            model=model,
            device=resolved_device,
            posthoc_bottleneck=posthoc_bottleneck is not None,
        )

    @torch.no_grad()
    def encode(
        self,
        audio: Tensor,
        *,
        normalize: bool = True,
        posthoc_bottleneck: bool | None = None,
        **kwargs: Any,
    ) -> Tensor:
        _, tokens = self.encode_latents(
            audio,
            normalize=normalize,
            posthoc_bottleneck=posthoc_bottleneck,
            **kwargs,
        )
        return tokens

    @torch.no_grad()
    def encode_latents(
        self,
        audio: Tensor,
        *,
        normalize: bool = True,
        posthoc_bottleneck: bool | None = None,
        **kwargs: Any,
    ) -> tuple[Tensor, Tensor]:
        if audio.dim() != 3:
            raise ValueError("StableCodec encode expects audio shape [batch, channels, time].")
        if audio.size(1) != NUM_CHANNELS:
            raise ValueError("StableCodec speech-16k expects mono audio with shape [batch, 1, time].")

        latents, tokens = self.model.encode(
            audio.to(self.device),
            posthoc_bottleneck=self._resolve_posthoc(posthoc_bottleneck),
            normalize=normalize,
            **kwargs,
        )
        return latents, tokens

    @torch.no_grad()
    def decode(
        self,
        tokens: Tensor,
        *,
        posthoc_bottleneck: bool | None = None,
        **kwargs: Any,
    ) -> Tensor:
        return self.model.decode(
            tokens.to(self.device),
            posthoc_bottleneck=self._resolve_posthoc(posthoc_bottleneck),
            **kwargs,
        )

    @torch.no_grad()
    def reconstruct(
        self,
        audio: Tensor,
        *,
        normalize: bool = True,
        posthoc_bottleneck: bool | None = None,
        **kwargs: Any,
    ) -> Tensor:
        tokens = self.encode(
            audio,
            normalize=normalize,
            posthoc_bottleneck=posthoc_bottleneck,
            **kwargs,
        )
        return self.decode(
            tokens,
            posthoc_bottleneck=posthoc_bottleneck,
            **kwargs,
        )

    def set_posthoc_bottleneck(self, stages: PosthocBottleneck) -> None:
        self.model.set_posthoc_bottleneck(stages)
        self.posthoc_bottleneck = True

    def _resolve_posthoc(self, value: bool | None) -> bool:
        if value is not None:
            return value
        return self.posthoc_bottleneck


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_stable_codec_model() -> Any:
    try:
        from stable_codec import StableCodec as UpstreamStableCodec
    except ImportError as exc:
        raise ImportError(
            "StableCodec requires the `stable-codec` package. Install "
            "`stable-codec` in a compatible environment before using this wrapper."
        ) from exc
    return UpstreamStableCodec


def _semantic_vocab_size(model: nn.Module) -> int:
    for name in ("semantic_vocab_size", "codebook_size", "cardinality"):
        value = getattr(model, name, None)
        if value is not None:
            return int(value)
    bottleneck = getattr(model, "bottleneck", None)
    if bottleneck is not None:
        value = getattr(bottleneck, "codebook_size", None)
        if value is not None:
            return int(value)
    return DEFAULT_SEMANTIC_VOCAB_SIZE
