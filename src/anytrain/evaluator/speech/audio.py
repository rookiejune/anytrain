from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor


def load_wave_batch(audio: Any, sample_rate: int) -> tuple[Tensor, int]:
    if isinstance(audio, str | Path):
        return _load_audio_file(Path(audio))

    sample_rate = validate_sample_rate(sample_rate)
    return coerce_wave_batch(audio), sample_rate


def coerce_wave_batch(audio: Any) -> Tensor:
    if isinstance(audio, torch.Tensor):
        wave = audio.detach()
    else:
        try:
            wave = torch.as_tensor(audio)
        except Exception as exc:
            raise TypeError(
                "audio must be a Tensor-like waveform or an audio file path."
            ) from exc

    wave = wave.float() if not torch.is_floating_point(wave) else wave.to(dtype=torch.float32)

    if wave.ndim == 1:
        wave = wave.unsqueeze(0)
    elif wave.ndim == 2:
        pass
    elif wave.ndim == 3:
        if wave.shape[1] == 0:
            raise ValueError("audio channel dimension must not be empty.")
        wave = wave.mean(dim=1)
    else:
        raise ValueError("audio must have shape [time], [batch, time], or [batch, channel, time].")

    if wave.shape[0] == 0 or wave.shape[-1] == 0:
        raise ValueError("audio waveform must not be empty.")
    return wave.contiguous()


def resample_wave(wave: Tensor, sample_rate: int, target_sample_rate: int) -> Tensor:
    sample_rate = validate_sample_rate(sample_rate)
    target_sample_rate = validate_sample_rate(target_sample_rate)
    if sample_rate == target_sample_rate:
        return wave

    try:
        import torchaudio
    except ImportError as exc:
        raise ImportError(
            "Resampling speech evaluator audio requires torchaudio. "
            "Install speech dependencies with `pip install anytrain[speech]`."
        ) from exc

    return torchaudio.functional.resample(wave, sample_rate, target_sample_rate)


def validate_sample_rate(sample_rate: int) -> int:
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int):
        raise TypeError("sample_rate must be an integer.")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive.")
    return sample_rate


def _load_audio_file(path: Path) -> tuple[Tensor, int]:
    torchaudio_error: Exception | None = None
    try:
        return _load_audio_file_with_torchaudio(path)
    except Exception as exc:
        torchaudio_error = exc

    try:
        return _load_audio_file_with_soundfile(path)
    except Exception as soundfile_error:
        raise RuntimeError(
            "Failed to read speech evaluator audio file with torchaudio or soundfile. "
            "Install speech dependencies with `pip install anytrain[speech]`. "
            f"torchaudio error: {torchaudio_error}; soundfile error: {soundfile_error}"
        ) from soundfile_error


def _load_audio_file_with_torchaudio(path: Path) -> tuple[Tensor, int]:
    import torchaudio

    wave, sample_rate = torchaudio.load(str(path))
    sample_rate = validate_sample_rate(int(sample_rate))
    if wave.ndim != 2:
        raise ValueError("torchaudio.load(...) must return waveform shape [channel, time].")
    if wave.shape[0] == 0:
        raise ValueError("audio channel dimension must not be empty.")
    wave = wave.to(dtype=torch.float32).mean(dim=0, keepdim=True)
    if wave.shape[-1] == 0:
        raise ValueError("audio waveform must not be empty.")
    return wave.contiguous(), sample_rate


def _load_audio_file_with_soundfile(path: Path) -> tuple[Tensor, int]:
    import soundfile

    data, sample_rate = soundfile.read(str(path), dtype="float32", always_2d=True)
    sample_rate = validate_sample_rate(int(sample_rate))
    wave = torch.from_numpy(data).transpose(0, 1)
    if wave.shape[0] == 0:
        raise ValueError("audio channel dimension must not be empty.")
    wave = wave.mean(dim=0, keepdim=True)
    if wave.shape[-1] == 0:
        raise ValueError("audio waveform must not be empty.")
    return wave.contiguous(), sample_rate
