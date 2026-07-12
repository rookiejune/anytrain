# Codec

`anytrain.codec` 为 optional audio codec wrapper 提供统一的离散运行时契约。具体模型的
加载方式和依赖可以不同，但配置完成后的 codec 使用相同张量语义：

```python
from typing_extensions import Protocol
from torch import Tensor

class Codec(Protocol):
    sample_rate: int
    codebook_sizes: tuple[int, ...]

    def encode(self, audio: Tensor, sample_rate: int) -> Tensor: ...
    def decode(self, codes: Tensor) -> Tensor: ...
```

- `audio` 使用 `[batch, channels, time]`。
- `encode()` 接收输入 waveform 的真实采样率，wrapper 在需要时内部重采样。
- `codes` 是整数 Tensor，统一使用 `[batch, frame, codebook]`。
- `sample_rate` 表示 `decode()` 输出 waveform 的采样率。
- `codebook_sizes[k]` 是第 `k` 个码本的 local id 数量。
- `decode()` 接收 codec 实例配置的完整、有序 K 个码本；码本轴不能重排或用任意子集冒充。

公共层不区分 semantic / acoustic code。LongCat wrapper 内部仍按上游 decoder 的要求拆分
第 0 个码本和后续码本，但调用方只操作统一的 `codes`。Stable Codec 的 posthoc bottleneck、
LongCat decoder、UniCodec domain / bandwidth 和 DAC `n_quantizers` 在构造 codec 时固定，不进入公共
`encode()` / `decode()` 参数。

连续 latent 并不是所有 codec 都共有的边界，因此不属于 `Codec` protocol。具体 wrapper
可以继续提供 `encode_features()`、`codes_to_features()` 或 `decode_features()` 等扩展。

具体 wrapper：

- [Descript Audio Codec](dac.md)
- [LongCat Audio Codec](longcat.md)
- [Stable Codec](stable-codec.md)
- [UniCodec](unicodec.md)
