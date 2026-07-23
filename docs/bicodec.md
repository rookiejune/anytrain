# Spark-TTS BiCodec

`anytrain.codec.bicodec` 是对 SparkAudio Spark-TTS 中 BiCodec 的 optional 薄集成。它不复制
Spark-TTS 源码，也不在 import `anytrain.codec` 时导入 `sparktts`；只有调用
`BiCodec.from_pretrained()` 时才需要上游源码和依赖。

## 安装

Spark-TTS 当前仓库没有标准 Python package metadata，不能仅靠 pip extra 安装出
`sparktts` 包。推荐先克隆官方源码并加入 `PYTHONPATH`：

    git clone https://github.com/SparkAudio/Spark-TTS.git /path/to/Spark-TTS
    export PYTHONPATH=/path/to/Spark-TTS:$PYTHONPATH
    python -m pip install -e ".[bicodec]"

`bicodec` extra 只声明 wrapper 需要的通用依赖，例如 `einx`、`huggingface-hub`、
`transformers`、`torchaudio`、`safetensors` 和 `omegaconf`。如果 Spark-TTS 后续发布
installable package，再把 `sparktts` 依赖加入 extra。

## 使用

    from anytrain.codec.bicodec import BiCodec

    codec = BiCodec.from_pretrained(
        cache_dir="/path/to/model-cache/bicodec",
        device="cuda",
    )

    tokens = codec.encode(audio, sample_rate=48000)
    audio_out = codec.decode(tokens)

    semantic = tokens.semantic
    global_tokens = tokens.global_tokens
    audio_out = codec.detokenize(semantic, global_tokens)

`audio` 使用 `[batch, 1, time]` 单声道 waveform。wrapper 会重采样到 Spark-TTS 的
16 kHz，并用上游 Wav2Vec2 特征抽取器生成 BiCodec 所需的 `feat`。默认把同一段音频裁成
reference clip 来提取说话人全局 token；如果目标音频和参考音频不同，可以显式传入：

    tokens = codec.encode(
        target_audio,
        sample_rate=target_sample_rate,
        ref_audio=prompt_audio,
        ref_sample_rate=prompt_sample_rate,
    )

## Token 契约

Spark-TTS BiCodec 的主边界不是统一 codec protocol 的单个 codes Tensor，而是：

- `semantic`：随时间变化的语义 token，由 BiCodec quantizer 产生。
- `global_tokens`：reference audio 产生的全局说话人 token。

因此 `encode()` 返回 `BiCodecTokens` dataclass，`decode()` 接收该 dataclass。
如果下游需要把 token 交给 LLM，请分别保存 semantic 和 global token 的 shape、dtype
和来源音频；不要把 global token 拼进 frame/codebook 轴伪装成普通 codec codes。

## Checkpoint 与缓存

默认模型仓库是 `SparkAudio/Spark-TTS-0.5B`。`ensure_bicodec_assets()` 只下载 BiCodec、
`wav2vec2-large-xlsr-53` 和根 `config.yaml`，避免为 codec tokenization 拉取完整 LLM 权重。

缓存路径优先级：

1. 显式传入 `cache_dir=...`
2. `HF_HOME/bicodec`
3. 如果 `HF_HOME` 未设置，anytrain 会设置从 `ANYTRAIN_HOME` 派生的 Hugging Face 缓存，
   再使用 `HF_HOME/bicodec`

已有本地模型目录时可以跳过下载：

    codec = BiCodec.from_pretrained(
        model_dir="/path/to/Spark-TTS-0.5B",
        device="cuda",
        local_files_only=True,
    )
