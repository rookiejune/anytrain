from __future__ import annotations

from enum import StrEnum
from importlib.resources import files
from pathlib import Path


class ExampleAudio(StrEnum):
    VCTK = "vctk"
    COLOR_YOUR_NIGHT = "color_your_night"


def list_example_audio() -> tuple[ExampleAudio, ...]:
    return tuple(ExampleAudio)


def example_audio_path(name: ExampleAudio | str) -> Path:
    audio = _resolve_audio(name)
    resource = files("anytrain.example").joinpath(*_relative_parts(audio))
    return Path(resource)


def vctk_path() -> Path:
    return example_audio_path(ExampleAudio.VCTK)


def color_your_night_path() -> Path:
    return example_audio_path(ExampleAudio.COLOR_YOUR_NIGHT)


def _resolve_audio(name: ExampleAudio | str) -> ExampleAudio:
    if isinstance(name, ExampleAudio):
        return name
    if not isinstance(name, str):
        raise TypeError("name must be an ExampleAudio or string.")
    try:
        return ExampleAudio(name)
    except ValueError as exc:
        supported = ", ".join(item.value for item in ExampleAudio)
        raise ValueError(f"Unknown example audio {name!r}. Supported: {supported}.") from exc


def _relative_parts(name: ExampleAudio) -> tuple[str, ...]:
    match name:
        case ExampleAudio.VCTK:
            return ("assets", "speech", "p225_001_mic1.flac")
        case ExampleAudio.COLOR_YOUR_NIGHT:
            return ("assets", "music", "color_your_night.mp3")
