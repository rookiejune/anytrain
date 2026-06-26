from __future__ import annotations

from typing import Any

INSTALL_HINT = (
    "Install Moss TTS dependencies with `pip install anytrain[moss-tts]`."
)


def load_transformers_auto_model_class() -> type[Any]:
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise ImportError(
            "`anytrain.tts.moss` requires `transformers` to load remote-code "
            f"MOSS-TTS checkpoints. {INSTALL_HINT}"
        ) from exc
    return AutoModelForCausalLM
