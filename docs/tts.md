# TTS

`anytrain.tts` provides small inference adapters for text-to-speech backends. It
does not own datasets, waveform caches, manifest files, training steps, or a
default CLI.

The stable public boundary is:

```python
from anytrain.tts import TTSOptions
from anytrain.tts.moss import MossTTS

tts = MossTTS.from_pretrained(
    "OpenMOSS-Team/MOSS-TTS-Nano",
    trust_remote_code=True,
    torch_dtype="auto",
)
audio = tts.synthesize("hello")
```

`synthesize()` is the high-level `text -> waveform` entry point. Backends may
also expose the lower-level debug path:

```text
tokenize(text) -> TTSTokens
generate(tokens) -> TTSGeneration
decode(generation) -> TTSOutput
```

`TTSOutput.waveform` is a float tensor with shape `[channels, time]`.

MOSS-TTS checkpoints are loaded as Hugging Face remote-code models. Keyword
arguments passed to `from_pretrained()` are load-time options, while runtime
conditioning belongs in `TTSOptions(extra=...)`, `runtime_kwargs=...`, or direct
`synthesize()` keyword arguments:

```python
tts = MossTTS.from_pretrained(
    "OpenMOSS-Team/MOSS-TTS-Nano",
    trust_remote_code=True,
    runtime_kwargs={
        "audio_tokenizer_pretrained_name_or_path": "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano",
    },
)

audio = tts.synthesize(
    "你好",
    TTSOptions(seed=7),
    output_audio_path="outputs/moss.wav",
    prompt_audio_path="assets/audio/zh_1.wav",
)
```

When merging `TTSOptions`, explicitly passed fields override defaults even when
the value is `None`; omitted fields inherit the backend config.
