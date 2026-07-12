from __future__ import annotations

from torch import Tensor
from typing_extensions import Protocol


class Codec(Protocol):
    sample_rate: int
    codebook_sizes: tuple[int, ...]

    def encode(self, audio: Tensor, sample_rate: int) -> Tensor:
        """Encode [batch, channels, time] audio as [batch, frame, codebook] ids."""
        ...

    def decode(self, codes: Tensor) -> Tensor:
        """Decode [batch, frame, codebook] ids at ``self.sample_rate``."""
        ...


__all__ = ["Codec"]
