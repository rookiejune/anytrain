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
`torchaudio==2.4`，而 `anytrain` 当前核心依赖是 `torch>=2.8`。因此第一版不把
`stable-codec` 放进 `anytrain` extra 自动安装，避免 pip resolver 得到互相冲突的
环境。需要使用 Stable Codec 时，先在兼容环境里安装上游包；如果确认新版本放宽了
torch pin，再把依赖加入 `pyproject.toml`。

Stable Codec 官方 README 还说明当前模型依赖 FlashAttention，不建议 CPU 推理。

在 `121` 上已验证 `stable-codec==0.1.2` 可以在独立 `py39` 环境安装并导入：
`python==3.9.25`、`torch==2.4.0+cu121`、`torchaudio==2.4.0+cu121`。安装时
`pypesq==1.2.4` 需要旧构建链，先安装 `numpy==1.23.5`、`Cython`、
`pip==24.0`、`setuptools==59.8.0`，再执行：

```bash
CFLAGS="-fcommon" python -m pip install --no-build-isolation pypesq==1.2.4
CFLAGS="-fcommon" python -m pip install stable-codec==0.1.2
```

复旦环境的完整记录见 `docs/codec-envs.md`。

## 使用

```python
from anytrain.codec.stable_codec import StableCodec

codec = StableCodec.from_pretrained(
    version="speech-16k",
    device="cuda",
)

codes = codec.encode(audio, sample_rate=16000)
audio_out = codec.decode(codes)

latents, codes = codec.encode_latents(audio, sample_rate=16000)
```

`audio` 需要是 `[batch, 1, time]` 的 16 kHz 单声道 waveform。Stable Codec 没有
UniCodec 的 `domain` 参数；默认模型是 `stabilityai/stable-codec-speech-16k`。

主路径按统一 codec 契约设计：`encode()` 返回 `[batch, frame, codebook]` codes，
`decode()` 接收同样形状的 codes
并重建 waveform。上游实现同时返回 pre-bottleneck continuous latents；需要直接操作这个
边界时使用 `encode_latents()`。

未启用 posthoc bottleneck 时，默认 speech 模型使用六维、每维 17 级的训练期 FSQ，
因此 native `codebook_sizes` 是 `(17^6,)`，即 `(24137569,)`，不是 `(46656,)`。
wrapper 从上游 `model.bottleneck.quantizer` 读取真实 `codebook_size` 和
`num_codebooks`，不为本地 config 猜测码本大小。

`121` 上确认的上游 `0.1.2` 签名与 wrapper 对齐：

- `StableCodec(pretrained_model=..., device=...)`
- `StableCodec(model_config_path=..., ckpt_path=..., device=...)`
- `encode(audio, posthoc_bottleneck=False, normalize=True, **kwargs)`
- `decode(tokens, posthoc_bottleneck=False, **kwargs)`
- `set_posthoc_bottleneck(stages)`

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
codes = codec.encode(audio, sample_rate=16000)
audio_out = codec.decode(codes)
```

支持的上游 preset：

- `"1x46656_400bps"`
- `"2x15625_700bps"`
- `"4x729_1000bps"`

`stable-codec==0.1.2` 的 posthoc encode 为每个 stage 返回一个 `[batch, frame, 1]`
Tensor 的 list。wrapper 会沿最后一维拼接为公共 `[batch, frame, codebook]` Tensor；
decode 时再拆回上游所需的 list。`46656` 仅是 `1x46656_400bps` preset 的码本大小，
需要把 Stable Codec tokens 交给语言模型时应显式选择 posthoc preset，并让产物配置记录
该 preset。

如果需要本地 config 和 checkpoint：

```python
codec = StableCodec.from_config(
    "config.json",
    ckpt_path="model.ckpt",
    device="cuda",
)
```
