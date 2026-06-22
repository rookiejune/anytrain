from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import torch
from torch import Tensor, nn


class _SpectrogramFactory(Protocol):
    def __call__(
        self,
        *,
        n_fft: int,
        hop_length: int | None,
        win_length: int | None,
        window_fn: Callable[..., Tensor],
        power: float | None,
        center: bool,
        normalized: bool,
    ) -> nn.Module: ...


class _MelSpectrogramFactory(Protocol):
    def __call__(
        self,
        *,
        sample_rate: int,
        n_fft: int,
        n_mels: int,
        hop_length: int | None,
        win_length: int | None,
        f_min: float,
        f_max: float,
        power: float,
        mel_scale: str,
    ) -> nn.Module: ...


TorchAudioMelSpectrogram: _MelSpectrogramFactory | None
TorchAudioSpectrogram: _SpectrogramFactory | None
try:
    from torchaudio.transforms import MelSpectrogram as _TorchAudioMelSpectrogram
    from torchaudio.transforms import Spectrogram as _TorchAudioSpectrogram
except ImportError:  # pragma: no cover - depends on optional local environment
    TorchAudioMelSpectrogram = None
    TorchAudioSpectrogram = None
else:
    TorchAudioMelSpectrogram = _TorchAudioMelSpectrogram
    TorchAudioSpectrogram = _TorchAudioSpectrogram


def _resolve_backend(backend: str, torchaudio_cls: object | None) -> str:
    if backend not in {"auto", "torchaudio", "torch"}:
        raise ValueError("backend must be 'auto', 'torchaudio', or 'torch'.")
    if backend == "torchaudio":
        if torchaudio_cls is None:
            raise ImportError("torchaudio is required for backend='torchaudio'.")
        return "torchaudio"
    if backend == "auto":
        return "torchaudio" if torchaudio_cls is not None else "torch"
    return "torch"


def _require_torchaudio_spectrogram() -> _SpectrogramFactory:
    if TorchAudioSpectrogram is None:
        raise ImportError("torchaudio is required for backend='torchaudio'.")
    return TorchAudioSpectrogram


def _require_torchaudio_mel_spectrogram() -> _MelSpectrogramFactory:
    if TorchAudioMelSpectrogram is None:
        raise ImportError("torchaudio is required for backend='torchaudio'.")
    return TorchAudioMelSpectrogram


class STFTTransform(nn.Module):
    window_tensor: Tensor

    def __init__(
        self,
        *,
        n_fft: int = 2048,
        hop_length: int | None = None,
        win_length: int | None = None,
        window: str = "hann",
        center: bool = True,
        normalized: bool = False,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft
        self.window = window
        self.center = center
        self.normalized = normalized
        self.backend = _resolve_backend(backend, TorchAudioSpectrogram)
        self.transform: nn.Module | None = None
        if self.backend == "torchaudio":
            spectrogram_cls = _require_torchaudio_spectrogram()
            self.transform = spectrogram_cls(
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window_fn=self._get_window_fn(window),
                power=None,
                center=center,
                normalized=normalized,
            )
        else:
            self.window_tensor = nn.Buffer(self._create_window(window, self.win_length))

    def forward(self, waveform: Tensor) -> Tensor:
        if self.backend == "torchaudio":
            if self.transform is None:
                raise RuntimeError("torchaudio spectrogram transform is not initialized.")
            return self.transform(waveform)

        original_shape = waveform.shape
        if waveform.ndim < 2:
            raise ValueError("waveform must include batch and time dimensions.")

        flattened = waveform.reshape(-1, original_shape[-1])
        spectrum = torch.stft(
            flattened,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window_tensor.to(device=waveform.device, dtype=waveform.dtype),
            center=self.center,
            normalized=self.normalized,
            return_complex=True,
        )
        return spectrum.reshape(*original_shape[:-1], *spectrum.shape[-2:])

    def _create_window(self, window: str, win_length: int) -> Tensor:
        return self._get_window_fn(window)(win_length)

    def _get_window_fn(self, window: str):
        match window:
            case "hann":
                return torch.hann_window
            case "none":
                return torch.ones
            case _:
                raise ValueError(f"Unsupported window type {window!r}.")


class MelSpectrogramTransform(nn.Module):
    mel_filter: Tensor

    def __init__(
        self,
        *,
        sample_rate: int = 44100,
        n_fft: int = 2048,
        n_mels: int = 128,
        hop_length: int | None = None,
        win_length: int | None = None,
        f_min: float = 0.0,
        f_max: float | None = None,
        power: float = 1.0,
        mel_scale: str = "htk",
        backend: str = "auto",
    ) -> None:
        super().__init__()
        if n_mels <= 0:
            raise ValueError("n_mels must be positive.")
        if power <= 0:
            raise ValueError("power must be positive.")
        if mel_scale not in {"htk", "slaney"}:
            raise ValueError("mel_scale must be 'htk' or 'slaney'.")

        self.backend = _resolve_backend(backend, TorchAudioMelSpectrogram)
        self.mel_scale = mel_scale
        self.power = power
        self.transform: nn.Module | None = None
        self.stft: STFTTransform | None = None
        f_max = f_max or sample_rate / 2
        if self.backend == "torchaudio":
            mel_spectrogram_cls = _require_torchaudio_mel_spectrogram()
            self.transform = mel_spectrogram_cls(
                sample_rate=sample_rate,
                n_fft=n_fft,
                n_mels=n_mels,
                hop_length=hop_length,
                win_length=win_length,
                f_min=f_min,
                f_max=f_max,
                power=power,
                mel_scale=mel_scale,
            )
        else:
            self.stft = STFTTransform(
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                backend="torch",
            )
            self.mel_filter = nn.Buffer(
                self._create_mel_filter(
                    sample_rate=sample_rate,
                    n_fft=n_fft,
                    n_mels=n_mels,
                    f_min=f_min,
                    f_max=f_max,
                    mel_scale=mel_scale,
                ),
            )

    def forward(self, waveform: Tensor) -> Tensor:
        if self.backend == "torchaudio":
            if self.transform is None:
                raise RuntimeError("torchaudio mel transform is not initialized.")
            return self.transform(waveform)

        if self.stft is None:
            raise RuntimeError("torch STFT fallback is not initialized.")
        magnitude = self.stft(waveform).abs().pow(self.power)
        return torch.einsum(
            "mf,...ft->...mt",
            self.mel_filter.to(device=waveform.device, dtype=waveform.dtype),
            magnitude,
        )

    def _create_mel_filter(
        self,
        *,
        sample_rate: int,
        n_fft: int,
        n_mels: int,
        f_min: float,
        f_max: float,
        mel_scale: str,
    ) -> Tensor:
        if not 0 <= f_min < f_max <= sample_rate / 2:
            raise ValueError("f_min and f_max must satisfy 0 <= f_min < f_max <= sample_rate / 2.")

        mel_points = torch.linspace(
            self._hz_to_mel(torch.tensor(float(f_min)), mel_scale=mel_scale),
            self._hz_to_mel(torch.tensor(float(f_max)), mel_scale=mel_scale),
            n_mels + 2,
        )
        hz_points = self._mel_to_hz(mel_points, mel_scale=mel_scale)
        fft_frequencies = torch.linspace(0, sample_rate / 2, n_fft // 2 + 1)

        filter_bank = torch.zeros(n_mels, n_fft // 2 + 1)
        for mel_index in range(n_mels):
            lower = hz_points[mel_index]
            center = hz_points[mel_index + 1]
            upper = hz_points[mel_index + 2]

            lower_slope = (fft_frequencies - lower) / (center - lower).clamp_min(1e-12)
            upper_slope = (upper - fft_frequencies) / (upper - center).clamp_min(1e-12)
            filter_bank[mel_index] = torch.minimum(lower_slope, upper_slope).clamp_min(0)
        return filter_bank

    @staticmethod
    def _hz_to_mel(frequency: Tensor, *, mel_scale: str) -> Tensor:
        if mel_scale == "htk":
            return 2595 * torch.log10(1 + frequency / 700)

        linear_scale = 200.0 / 3.0
        min_log_hz = 1000.0
        min_log_mel = min_log_hz / linear_scale
        log_step = (
            torch.log(torch.tensor(6.4, device=frequency.device, dtype=frequency.dtype)) / 27.0
        )
        linear_mel = frequency / linear_scale
        log_mel = min_log_mel + torch.log(frequency / min_log_hz) / log_step
        return torch.where(frequency >= min_log_hz, log_mel, linear_mel)

    @staticmethod
    def _mel_to_hz(mel: Tensor, *, mel_scale: str) -> Tensor:
        if mel_scale == "htk":
            return 700 * (10 ** (mel / 2595) - 1)

        linear_scale = 200.0 / 3.0
        min_log_hz = 1000.0
        min_log_mel = min_log_hz / linear_scale
        log_step = torch.log(torch.tensor(6.4, device=mel.device, dtype=mel.dtype)) / 27.0
        linear_hz = mel * linear_scale
        log_hz = min_log_hz * torch.exp(log_step * (mel - min_log_mel))
        return torch.where(mel >= min_log_mel, log_hz, linear_hz)
