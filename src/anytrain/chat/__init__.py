"""Environment-backed stateful chat and image helpers for optional LLM providers."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum, auto
from typing import Any, Literal, NamedTuple, TypedDict

DEEPSEEK_BASE_URL_ENV = "DEEPSEEK_BASE_URL"
DEEPSEEK_MODEL_ENV = "DEEPSEEK_MODEL"
DEEPSEEK_API_KEY_ENV = "DEEPSEEK_API_KEY"
GLM_BASE_URL_ENV = "GLM_BASE_URL"
GLM_MODEL_ENV = "GLM_MODEL"
GLM_API_KEY_ENV = "GLM_API_KEY"
DEEPSEEK_REASONING_EFFORT = "high"
DEEPSEEK_SYSTEM_PROMPT = "You are a helpful assistant"
GLM_MAX_TOKENS = 65536
GLM_TEMPERATURE = 1.0
GLM_IMAGE_MODEL = "glm-image"
GLM_IMAGE_SIZE = "1280x1280"
GLM_IMAGE_TIMEOUT = 120.0
GLM_IMAGE_GENERATIONS_PATH = "/images/generations"
INSTALL_HINT = "Install chat dependencies with `pip install anytrain[chat]`."


class ModelType(StrEnum):
    DEEPSEEK = auto()
    GLM = auto()


class _EnvNames(NamedTuple):
    base_url: str
    model: str
    api_key: str


class ChatMessage(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


class ImageOutput(NamedTuple):
    url: str | None
    b64_json: str | None
    revised_prompt: str | None


class ImageGeneration(NamedTuple):
    model: str
    prompt: str
    size: str
    data: tuple[ImageOutput, ...]
    raw: Mapping[str, object]

    @property
    def url(self) -> str | None:
        return self.data[0].url

    @property
    def b64_json(self) -> str | None:
        return self.data[0].b64_json


@dataclass(frozen=True)
class ChatConfig:
    base_url: str
    model: str
    api_key: str = field(repr=False)


_ENV_NAMES: dict[ModelType, _EnvNames] = {
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


class Chat:
    def __init__(self, model_type: ModelType | str = ModelType.DEEPSEEK):
        self.model_type = ModelType(model_type)
        self._config = config_from_env(self.model_type)
        self._client = _create_client(self.model_type, self._config)
        self._messages = _initial_messages(self.model_type)

    @property
    def base_url(self) -> str:
        return self._config.base_url

    @property
    def model(self) -> str:
        return self._config.model

    def refresh(self) -> None:
        self._messages = _initial_messages(self.model_type)

    def __call__(self, prompt: str, *, refresh: bool = False) -> str:
        if prompt == "":
            raise ValueError("prompt must not be empty.")
        if refresh:
            self.refresh()
        messages: list[ChatMessage] = [
            *self._messages,
            {"role": "user", "content": prompt},
        ]
        response = _request(self.model_type, self._client, self._config.model, messages)
        self._messages = [
            *messages,
            {"role": "assistant", "content": response},
        ]
        return response

    def image(
        self,
        prompt: str,
        *,
        size: str = GLM_IMAGE_SIZE,
        model: str = GLM_IMAGE_MODEL,
        timeout: float | None = GLM_IMAGE_TIMEOUT,
    ) -> ImageGeneration:
        if prompt == "":
            raise ValueError("prompt must not be empty.")
        if size == "":
            raise ValueError("size must not be empty.")
        if model == "":
            raise ValueError("model must not be empty.")
        return _generate_image(
            self.model_type,
            self._config,
            prompt=prompt,
            model=model,
            size=size,
            timeout=timeout,
        )


def config_from_env(model_type: ModelType | str) -> ChatConfig:
    names = _ENV_NAMES[ModelType(model_type)]
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


def _create_client(model_type: ModelType, config: ChatConfig) -> object:
    if model_type == ModelType.GLM:
        return _create_glm_client(config)
    if model_type == ModelType.DEEPSEEK:
        return _create_deepseek_client(config)
    raise AssertionError(f"Unhandled model_type: {model_type!r}")


def _create_deepseek_client(config: ChatConfig) -> object:
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - exercised in environments without the extra.
        raise ImportError(
            f"`anytrain.chat` DeepSeek backend requires `openai`. {INSTALL_HINT}"
        ) from exc

    return OpenAI(api_key=config.api_key, base_url=config.base_url)


def _create_glm_client(config: ChatConfig) -> object:
    try:
        from zai import ZhipuAiClient
    except ImportError as exc:  # pragma: no cover - exercised in environments without the extra.
        raise ImportError(
            f"`anytrain.chat` GLM backend requires `zai-sdk`. {INSTALL_HINT}"
        ) from exc

    return ZhipuAiClient(api_key=config.api_key, base_url=config.base_url)


def _initial_messages(model_type: ModelType) -> list[ChatMessage]:
    if model_type == ModelType.DEEPSEEK:
        return [{"role": "system", "content": DEEPSEEK_SYSTEM_PROMPT}]
    if model_type == ModelType.GLM:
        return []
    raise AssertionError(f"Unhandled model_type: {model_type!r}")


def _request(
    model_type: ModelType,
    client: object,
    model: str,
    messages: Sequence[ChatMessage],
) -> str:
    if model_type == ModelType.GLM:
        return _request_glm(client, model, messages)
    if model_type == ModelType.DEEPSEEK:
        return _request_deepseek(client, model, messages)
    raise AssertionError(f"Unhandled model_type: {model_type!r}")


def _request_deepseek(
    client: object,
    model: str,
    messages: Sequence[ChatMessage],
) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=False,
        reasoning_effort=DEEPSEEK_REASONING_EFFORT,
        extra_body={"thinking": {"type": "enabled"}},
    )
    return _response_text(response, provider="DeepSeek")


def _request_glm(
    client: object,
    model: str,
    messages: Sequence[ChatMessage],
) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        thinking={"type": "enabled"},
        max_tokens=GLM_MAX_TOKENS,
        temperature=GLM_TEMPERATURE,
    )
    return _response_text(response, provider="GLM")


def _generate_image(
    model_type: ModelType,
    config: ChatConfig,
    *,
    prompt: str,
    model: str,
    size: str,
    timeout: float | None,
) -> ImageGeneration:
    if model_type == ModelType.GLM:
        return _generate_glm_image(
            config,
            prompt=prompt,
            model=model,
            size=size,
            timeout=timeout,
        )
    if model_type == ModelType.DEEPSEEK:
        raise NotImplementedError("DeepSeek image generation is not supported.")
    raise AssertionError(f"Unhandled model_type: {model_type!r}")


def _generate_glm_image(
    config: ChatConfig,
    *,
    prompt: str,
    model: str,
    size: str,
    timeout: float | None,
) -> ImageGeneration:
    requests = _load_requests()
    response = requests.post(
        url=_glm_image_url(config.base_url),
        json={
            "model": model,
            "prompt": prompt,
            "size": size,
        },
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        },
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, Mapping):
        raise TypeError("GLM image response must be a JSON object.")
    data = _image_outputs(payload)
    return ImageGeneration(
        model=model,
        prompt=prompt,
        size=size,
        data=data,
        raw=dict(payload),
    )


def _load_requests() -> Any:
    try:
        import requests
    except ImportError as exc:  # pragma: no cover - exercised in environments without the extra.
        raise ImportError(
            f"`anytrain.chat` GLM image generation requires `requests`. {INSTALL_HINT}"
        ) from exc
    return requests


def _glm_image_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}{GLM_IMAGE_GENERATIONS_PATH}"


def _image_outputs(payload: Mapping[str, object]) -> tuple[ImageOutput, ...]:
    value = payload.get("data")
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError("GLM image response must contain a data sequence.")
    if len(value) == 0:
        raise ValueError("GLM image response data must not be empty.")
    return tuple(_image_output(item) for item in value)


def _image_output(item: object) -> ImageOutput:
    if not isinstance(item, Mapping):
        raise TypeError("GLM image response data items must be JSON objects.")
    url = _optional_str(item, "url")
    b64_json = _optional_str(item, "b64_json")
    revised_prompt = _optional_str(item, "revised_prompt")
    if url is None and b64_json is None:
        raise TypeError("GLM image response data item must contain url or b64_json.")
    return ImageOutput(
        url=url,
        b64_json=b64_json,
        revised_prompt=revised_prompt,
    )


def _optional_str(payload: Mapping[object, object], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"GLM image response field {key!r} must be a string.")
    return value


def _response_text(response: object, *, provider: str) -> str:
    try:
        message = response.choices[0].message
    except (AttributeError, IndexError, TypeError) as exc:
        raise TypeError(f"{provider} response must contain choices[0].message.") from exc

    if isinstance(message, str):
        return message
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
    raise TypeError(f"{provider} response message must be a string or contain string content.")


__all__ = ["Chat"]
