from __future__ import annotations

from typing import Any

INSTALL_HINT = (
    "Install Qwen3-TTS with "
    "`python -m pip install git+https://github.com/QwenLM/Qwen3-TTS.git`."
)


def load_qwen3_tts_model_class() -> type[Any]:
    try:
        from qwen_tts import Qwen3TTSModel
    except ImportError as exc:
        raise ImportError(
            "anytrain.tts.qwen requires Qwen3-TTS to load Qwen3-TTS "
            f"checkpoints. {INSTALL_HINT}"
        ) from exc
    return Qwen3TTSModel
