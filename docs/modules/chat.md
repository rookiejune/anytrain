# `anytrain.chat` Design

## 定位

`anytrain.chat` 提供环境变量驱动的轻量有状态大模型调用入口，用于训练工程里的实验总结、日志解释和 LLM 辅助评估原型。它不是训练入口、配置系统或 provider SDK 封装层；下游项目仍然负责决定何时调用、如何记录结果和如何处理失败。

当前公开接口：

```python
from anytrain import chat

response = chat.Chat("deepseek")("summarize this training curve")
```

`anytrain.chat` 导出 `Chat`、公共配置、provider 枚举和返回类型。环境变量名与 provider 默认参数仍是模块实现细节，不进入 `__all__`。

当前 `__all__`：

- `Chat`
- `ChatConfig`
- `ImageGeneration`
- `ImageOutput`
- `ModelType`
- `config_from_env`

同一个 `Chat` 实例会保留已成功请求的 user/assistant 消息，后续请求会把历史消息作为上下文继续发送。需要开启新上下文时，可以调用 `client.refresh()`，或在单次请求中传入 `refresh=True`：

```python
client = chat.Chat("deepseek")
client("summarize epoch 1")
client("compare it with epoch 2")
client.refresh()
client("start a new summary")
client("ignore previous context", refresh=True)
```

## 当前实现

当前模块实现稳定外部契约、实例内消息上下文、环境变量解析、DeepSeek OpenAI-compatible SDK 调用、GLM SDK 调用和 GLM 图片生成调用。缺少对应 optional 依赖时会明确抛出 `ImportError`，避免静默 fallback。

支持的 `model_type`：

- `deepseek`
- `glm`

环境变量：

- `DEEPSEEK_BASE_URL`
- `DEEPSEEK_MODEL`
- `DEEPSEEK_API_KEY`
- `GLM_BASE_URL`
- `GLM_MODEL`
- `GLM_API_KEY`

这些变量缺失或为空时，`Chat(...)` 会直接抛出 `ValueError`。

## Optional Extra

provider 请求实现放在 `chat` extra 后面，不进入默认依赖：

```bash
python -m pip install "anytrain[chat]"
```

当前 extra 依赖 `openai`、`requests` 和 `zai-sdk`。根包 `import anytrain` 不导入 chat backend，也不导出 `Chat`。

DeepSeek 调用使用：

- `from openai import OpenAI`
- `OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)`
- system prompt: `You are a helpful assistant`
- `stream=False`
- `reasoning_effort="high"`
- `extra_body={"thinking": {"type": "enabled"}}`

GLM 调用使用：

- `from zai import ZhipuAiClient`
- `ZhipuAiClient(api_key=GLM_API_KEY, base_url=GLM_BASE_URL)`
- `thinking={"type": "enabled"}`
- `max_tokens=65536`
- `temperature=1.0`

`GLM_BASE_URL` 应填写 SDK 根路径，例如 `https://open.bigmodel.cn/api/paas/v4`；`zai-sdk` 会自行拼接 `/chat/completions`。

GLM 图片生成使用同步 HTTP 调用：

```python
image = chat.Chat("glm").image(
    "一只可爱的小猫咪，坐在阳光明媚的窗台上，背景是蓝天白云.",
    size="1280x1280",
)
```

- 默认模型：`glm-image`
- 默认尺寸：`1280x1280`
- URL：`GLM_BASE_URL` 拼接 `/images/generations`
- 返回值包含 `data`、`raw`，并提供首张图片的 `url` 和 `b64_json` 快捷属性

DeepSeek 当前不支持图片生成，调用 `Chat("deepseek").image(...)` 会抛出
`NotImplementedError`。

## 边界

`anytrain.chat` 不做：

- 不接管 prompt 模板、任务 schema、provider cache 观测或结果记录。
- 不在缺失环境变量时静默换 provider。
- 不把 API key 写入 repr、日志或异常文本。
- 不在 package root 导出 `Chat`。
- 不为图片生成维护额外对话上下文；图片生成是独立请求。

后续扩展 provider 参数时，应保持 `Chat(model_type)(prompt, refresh=False) -> str` 的顶层契约，provider-specific 参数再通过明确配置扩展。
