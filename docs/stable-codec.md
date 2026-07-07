# Stable Codec

`anytrain.codec.stable_codec` 是对 Stability AI `stable-codec` 包的 optional
薄集成层。它不复制模型源码，不进入 core import；只有调用
`StableCodec.from_pretrained()` 或 `from_config()` 时才导入上游
`stable_codec`。

## 安装

官方安装方式：

```bash
python -m pip install stable-codec
python -m pip install -U flash-attn --no-build-isolation
```

当前官方 `stable-codec==0.1.2` 的 `setup.py` 硬性依赖 `torch==2.4` 和
`torchaudio==2.4`，而 `anytrain` 当前核心依赖是 `torch>=2.12`。因此第一版不把
`stable-codec` 放进 `anytrain` extra 自动安装，避免 pip resolver 得到互相冲突的
环境。需要使用 Stable Codec 时，先在兼容环境里安装上游包；如果确认新版本放宽了
torch pin，再把依赖加入 `pyproject.toml`。

Stable Codec 官方 README 还说明当前模型依赖 FlashAttention，不建议 CPU 推理。

## 使用

```python
from anytrain.codec.stable_codec import StableCodec

codec = StableCodec.from_pretrained(
    version="speech-16k",
    device="cuda",
)

tokens = codec.encode(audio)
audio_out = codec.decode(tokens)

latents, tokens = codec.encode_latents(audio)
```

`audio` 需要是 `[batch, 1, time]` 的 16 kHz 单声道 waveform。Stable Codec 没有
UniCodec 的 `domain` 参数；默认模型是 `stabilityai/stable-codec-speech-16k`。

主路径按离散 token roundtrip 设计：`encode()` 返回 tokens，`decode()` 接收 tokens
并重建 waveform。上游实现同时返回 pre-bottleneck continuous latents；需要直接操作这个
边界时使用 `encode_latents()`。

## Posthoc Bottleneck

上游推荐可选 posthoc FSQ bottleneck。初始化时可以直接启用：

```python
codec = StableCodec.from_pretrained(
    device="cuda",
    posthoc_bottleneck="2x15625_700bps",
)
```

也可以先加载模型，再设置：

```python
codec.set_posthoc_bottleneck("2x15625_700bps")
tokens = codec.encode(audio)
audio_out = codec.decode(tokens)
```

支持的上游 preset：

- `"1x46656_400bps"`
- `"2x15625_700bps"`
- `"4x729_1000bps"`

如果需要本地 config 和 checkpoint：

```python
codec = StableCodec.from_config(
    "config.json",
    ckpt_path="model.ckpt",
    device="cuda",
)
```
