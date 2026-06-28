# TTS

`anytrain.tts` provides small inference adapters for text-to-speech backends. It
does not own datasets, waveform caches, manifest files, training steps, or a
default CLI.

The stable public boundary is:

```python
from anytrain.tts import TTSOptions
from anytrain.tts.moss import MossTTS

tts = MossTTS.from_pretrained(
    "OpenMOSS-Team/MOSS-TTS-v1.5",
    trust_remote_code=True,
    torch_dtype="auto",
)
audio = tts.synthesize("hello")
```

`synthesize()` is the high-level `text -> waveform` entry point.
`TTSOutput.waveform` is a float tensor with shape `[channels, time]`.

The MOSS adapter intentionally targets only MOSS-TTS v1.5. It loads the
Hugging Face remote-code model and processor, then runs the v1.5 processor
generation path. Keyword arguments passed to `from_pretrained()` are load-time
options, while runtime conditioning belongs in `TTSOptions(extra=...)`,
`runtime_kwargs=...`, or direct `synthesize()` keyword arguments:

```python
tts = MossTTS.from_pretrained(
    "OpenMOSS-Team/MOSS-TTS-v1.5",
    trust_remote_code=True,
    codec_model="OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano",
)

audio = tts.synthesize(
    "你好",
    TTSOptions(seed=7),
    prompt_audio_path="assets/audio/zh_1.wav",
)
```

The v1.5 adapter returns audio in memory and does not support legacy
`output_audio_path`, `speaker`, `model.inference()`, or
`tokenize -> generate -> decode` compatibility paths.

When merging `TTSOptions`, explicitly passed fields override defaults even when
the value is `None`; omitted fields inherit the backend config.
