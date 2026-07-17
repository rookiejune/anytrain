from __future__ import annotations

import os
from typing import NamedTuple

from .types import ModelType

DEEPSEEK_BASE_URL_ENV = "DEEPSEEK_BASE_URL"
DEEPSEEK_MODEL_ENV = "DEEPSEEK_MODEL"
DEEPSEEK_API_KEY_ENV = "DEEPSEEK_API_KEY"
GLM_BASE_URL_ENV = "GLM_BASE_URL"
GLM_MODEL_ENV = "GLM_MODEL"
GLM_API_KEY_ENV = "GLM_API_KEY"


class ChatConfig:
    __slots__ = ("api_key", "base_url", "model")

    def __init__(self, base_url: str, model: str, api_key: str) -> None:
        self.base_url = base_url
        self.model = model
        self.api_key = api_key

    def __repr__(self) -> str:
        return f"ChatConfig(base_url={self.base_url!r}, model={self.model!r}, api_key=<hidden>)"


class _EnvNames(NamedTuple):
    base_url: str
    model: str
    api_key: str


ENV_NAMES: dict[ModelType, _EnvNames] = {
    ModelType.DEEPSEEK: _EnvNames(
        base_url=DEEPSEEK_BASE_URL_ENV,
        model=DEEPSEEK_MODEL_ENV,
        api_key=DEEPSEEK_API_KEY_ENV,
    ),
    ModelType.GLM: _EnvNames(
        base_url=GLM_BASE_URL_ENV,
        model=GLM_MODEL_ENV,
        api_key=GLM_API_KEY_ENV,
    ),
}


def config_from_env(model_type: ModelType | str) -> ChatConfig:
    names = ENV_NAMES[ModelType(model_type)]
    return ChatConfig(
        base_url=_required_env(names.base_url),
        model=_required_env(names.model),
        api_key=_required_env(names.api_key),
    )


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if value is None:
        raise ValueError(f"{name} must be set for anytrain.chat.")
    if value == "":
        raise ValueError(f"{name} must not be empty.")
    return value
