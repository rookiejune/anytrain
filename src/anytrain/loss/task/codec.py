from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor

from anytrain._compat import StrEnum

from ..balancer import FixedWeightLossBalancer, LossBalancerABC
from ..group import LossGroup
from ..spectral import MultiScaleMelLoss, MultiScaleSTFTLoss
from ..temporal import SDRLoss

DAC_MEL_N_FFT_LIST = (2048, 1024, 512, 256, 128, 64, 32)
DAC_MEL_N_MELS_LIST = (320, 160, 80, 40, 20, 10, 5)
DYNACODEC_STFT_N_FFT_LIST = (2048, 1024, 512, 256)
DYNACODEC_LOSS_WEIGHTS = {
    "si_sdr": 1.0,
    "multi_mel": 10.0,
    "multi_stft": 10.0,
}


class CodecLossPreset(StrEnum):
    DAC = "dac"
    DYNACODEC = "dynacodec"


class CodecLoss(LossGroup):
    def forward(
        self,
        input: Tensor,
        target: Tensor,
        *,
        lengths: Tensor | Sequence[int] | None = None,
    ):
        if lengths is None:
            return super().forward(input, target)
        return self._forward_with_lengths(input, target, lengths=lengths)

    @classmethod
    def from_preset(
        cls,
        preset: CodecLossPreset | str,
        *,
        sample_rate: int = 44100,
        mel_n_fft_list: Sequence[int] = DAC_MEL_N_FFT_LIST,
        mel_n_mels_list: Sequence[int] = DAC_MEL_N_MELS_LIST,
        stft_n_fft_list: Sequence[int] = DYNACODEC_STFT_N_FFT_LIST,
        mel_loss_kwargs: Mapping[str, Any] | None = None,
        stft_loss_kwargs: Mapping[str, Any] | None = None,
        balancer: LossBalancerABC | None = None,
        mel_scale: str = "htk",
        backend: str = "auto",
    ) -> CodecLoss:
        preset = cls._resolve_preset(preset)
        if preset == CodecLossPreset.DAC:
            return cls(
                {
                    "multi_mel": MultiScaleMelLoss(
                        sample_rate=sample_rate,
                        n_fft_list=mel_n_fft_list,
                        n_mels_list=mel_n_mels_list,
                        loss_kwargs=mel_loss_kwargs,
                        mel_scale=mel_scale,
                        backend=backend,
                    )
                },
                balancer=balancer,
            )
        if preset == CodecLossPreset.DYNACODEC:
            return cls(
                {
                    "si_sdr": SDRLoss(),
                    "multi_mel": MultiScaleMelLoss(
                        sample_rate=sample_rate,
                        n_fft_list=mel_n_fft_list,
                        n_mels_list=mel_n_mels_list,
                        loss_kwargs=mel_loss_kwargs,
                        mel_scale=mel_scale,
                        backend=backend,
                    ),
                    "multi_stft": MultiScaleSTFTLoss(
                        n_fft_list=stft_n_fft_list,
                        loss_kwargs=stft_loss_kwargs,
                        backend=backend,
                    ),
                },
                balancer=balancer or FixedWeightLossBalancer(DYNACODEC_LOSS_WEIGHTS),
            )
        raise ValueError(f"Unsupported codec loss preset {preset!r}.")

    @staticmethod
    def _resolve_preset(preset: CodecLossPreset | str) -> CodecLossPreset:
        if isinstance(preset, CodecLossPreset):
            return preset
        if not isinstance(preset, str):
            raise TypeError("preset must be a CodecLossPreset or string.")
        try:
            return CodecLossPreset(preset)
        except ValueError as exc:
            supported = ", ".join(item.value for item in CodecLossPreset)
            raise ValueError(
                f"Unknown codec loss preset {preset!r}. Supported presets: {supported}."
            ) from exc

    def _forward_with_lengths(
        self,
        input: Tensor,
        target: Tensor,
        *,
        lengths: Tensor | Sequence[int],
    ):
        _validate_audio_pair(input, target)
        length_values = _validate_lengths(lengths, batch_size=input.size(0), max_length=input.size(-1))
        total = input.new_zeros(())
        detail_sums: dict[str, Tensor] = {}
        for sample_length, batch_indices in _group_length_indices(length_values):
            group_input = input[batch_indices, ..., :sample_length]
            group_target = target[batch_indices, ..., :sample_length]
            group_total, group_details = super().forward(group_input, group_target)
            weight = input.new_tensor(float(len(batch_indices)))
            total = total + group_total * weight
            for name, value in group_details.items():
                value_tensor = _detail_to_tensor(value, device=input.device, dtype=input.dtype)
                detail_sums[name] = detail_sums.get(name, input.new_zeros(())) + value_tensor * weight

        divisor = input.new_tensor(float(len(length_values)))
        details = {name: value / divisor for name, value in detail_sums.items()}
        return total / divisor, details


def _validate_audio_pair(input: Tensor, target: Tensor) -> None:
    if input.shape != target.shape:
        raise ValueError("input and target must have the same shape when lengths are provided.")
    if input.ndim < 3:
        raise ValueError("audio tensors must have shape (batch, channels, time) or higher.")


def _validate_lengths(
    lengths: Tensor | Sequence[int],
    *,
    batch_size: int,
    max_length: int,
) -> tuple[int, ...]:
    if isinstance(lengths, Tensor):
        if lengths.ndim != 1:
            raise ValueError("lengths must be a 1D tensor or sequence.")
        raw_lengths = [int(value) for value in lengths.detach().cpu().tolist()]
    else:
        raw_lengths = [int(value) for value in lengths]

    if len(raw_lengths) != batch_size:
        raise ValueError(
            f"lengths must contain one value per batch item: got {len(raw_lengths)}, "
            f"expected {batch_size}."
        )
    for length in raw_lengths:
        if length <= 0:
            raise ValueError("lengths must be positive.")
        if length > max_length:
            raise ValueError(
                f"lengths cannot exceed audio time dimension: got {length}, max {max_length}."
            )
    return tuple(raw_lengths)


def _group_length_indices(lengths: Sequence[int]) -> tuple[tuple[int, list[int]], ...]:
    groups: dict[int, list[int]] = {}
    for index, length in enumerate(lengths):
        groups.setdefault(length, []).append(index)
    return tuple(groups.items())


def _detail_to_tensor(value: float | Tensor, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    if isinstance(value, Tensor):
        return value.to(device=device, dtype=dtype)
    return torch.tensor(float(value), device=device, dtype=dtype)
