from __future__ import annotations

from torch import Tensor


def resample(audio: Tensor, sample_rate: int, target_sample_rate: int) -> Tensor:
    if sample_rate == target_sample_rate:
        return audio

    try:
        from torchaudio.functional import resample as torchaudio_resample
    except ImportError as error:
        raise ImportError(
            "Codec input resampling requires torchaudio. Install `anytrain[audio]`."
        ) from error
    return torchaudio_resample(audio, sample_rate, target_sample_rate)
