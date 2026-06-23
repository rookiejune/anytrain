from __future__ import annotations

from enum import auto

from anytrain.types import AutoNameEnum


class QuantizerType(AutoNameEnum):
    AGRVQ = auto()
    FSQ = auto()
    GVQ = auto()
    VQ = auto()
    RVQ = auto()
