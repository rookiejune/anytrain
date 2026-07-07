# LongCat Audio Codec

`anytrain.codec.longcat` 是对
`meituan-longcat/LongCat-Audio-Codec` 的 optional 集成层。它不进入 core import，
只在用户显式安装 `longcat` extra 并调用相关 API 时使用。

## 安装

LongCat 源码维护在可安装 fork：

```bash
python -m pip install -e ../LongCat-Audio-Codec
```

fork 推送后，下游可以直接安装 anytrain extra：

```bash
python -m pip install -e ".[longcat]"
```

`longcat` extra 依赖：

- `longcat-audio-codec @ git+https://github.com/rookiejune/LongCat-Audio-Codec.git`
- `huggingface-hub`

## 缓存路径

checkpoint 自动下载到缓存目录。路径优先级：

1. 显式传入 `cache_dir=...`
2. `HF_HOME/longcat-audio-codec`
3. 如果 `HF_HOME` 未设置，anytrain 会设置
   `HF_HOME=${ANYTRAIN_HOME:-~/.anytrain}/huggingface`，再使用
   `$HF_HOME/longcat-audio-codec`

远程服务器上推荐：

```bash
export ANYTRAIN_HOME=/mnt/pami202/zhuyin/.anytrain
```

下载后的 LongCat 权重在 `$HF_HOME/longcat-audio-codec/ckpts`，生成的 patched
config 在 `$HF_HOME/longcat-audio-codec/configs`。

## 使用

```python
from anytrain.codec.longcat import LongCat

codec = LongCat.from_pretrained(
    device="cuda",
    decoders=("24k_4codebooks",),
)

semantic_codes, acoustic_codes = codec.encode(
    audio,
    sample_rate=16000,
    n_acoustic_codebooks=2,
)
audio_24k = codec.decode(
    semantic_codes,
    acoustic_codes,
    decoder="24k_4codebooks",
)

acoustic_features = codec.acoustic_codes_to_features(
    acoustic_codes,
    decoder="24k_4codebooks",
)
audio_from_features = codec.decode_features(
    semantic_codes,
    acoustic_features,
    decoder="24k_4codebooks",
)
```

`acoustic_codes_to_features()` 显式调用 LongCat decoder 的 acoustic dequantizer，
对外返回 `[batch, time, dim]` 的连续 acoustic features。`decode_features()` 接收同样
形状的连续 features，再交给 LongCat decoder 合成波形。下游 DiT 或 flow sampler 应该接
这个 feature 边界；只有原始 codec roundtrip 才直接使用离散 `acoustic_codes`。

默认从 Hugging Face 仓库 `meituan-longcat/LongCat-Audio-Codec` 下载：

- `LongCatAudioCodec_encoder.pt`
- `LongCatAudioCodec_encoder_cmvn.npy`
- `LongCatAudioCodec_decoder_16k_4codebooks.pt`
- `LongCatAudioCodec_decoder_24k_2codebooks.pt`
- `LongCatAudioCodec_decoder_24k_4codebooks.pt`

`local_files_only=True` 可以在离线环境中只使用已有缓存。
`decoders` 只会准备请求的 decoder；encoder 和 encoder cmvn 始终会准备，
缺失的请求项会按需从 Hugging Face 下载。
