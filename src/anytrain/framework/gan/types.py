from __future__ import annotations

from enum import auto

from anytrain._compat import StrEnum


class GAN(StrEnum):
    Hinge = auto()
    LSGAN = auto()
    WGAN = auto()


class Preset(StrEnum):
    DAC = auto()


class Reduction(StrEnum):
    Mean = auto()
    Sum = auto()


def _gan(value: GAN | str) -> GAN:
    if isinstance(value, GAN):
        return value
    if not isinstance(value, str):
        raise TypeError("gan must be a GAN or string.")
    try:
        return GAN(value)
    except ValueError as exc:
        supported = ", ".join(item.value for item in GAN)
        raise ValueError(f"Unknown GAN type {value!r}. Supported: {supported}.") from exc


def _preset(value: Preset) -> Preset:
    if not isinstance(value, Preset):
        raise TypeError("preset must be a Preset.")
    return value
