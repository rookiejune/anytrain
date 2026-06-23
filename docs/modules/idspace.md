# `anytrain.idspace` Design

## 定位

`anytrain.idspace` 管 unified token id space：同一套 layout 同时驱动多模态 tokenizer、
embedding 和 tied head。它不训练 tokenizer，不解释 batch schema，也不替下游修改完整 HF model。

当前公开对象：

- `Modality`
- `TokenLayout`
- `ModalityRange`
- `MultiTokenizer`
- `SubTokenizer` protocol
- `HFTokenizerAdapter`
- `TokenEmbedding`

## Layout

`TokenLayout` 是 tokenizer 和 embedding 共享的 id 空间定义。普通 token 的 global id
在不同 modality 之间不能重叠；special token 是全局共享的 sparse token id set，不属于
任何 modality。embedding 先匹配 special token id，剩下的 id 再按 modality range 路由：

```text
special_token_ids: {pad: 0, bos: 151643, eos: 151645}
text:    [0, text_vocab)
audio:   [text_vocab, text_vocab + audio_vocab)
```

`Modality` 是写死的 `StrEnum`，第一版只包含 `Modality.TEXT` 和 `Modality.AUDIO`。
layout、sub-tokenizer 和 embedding 构造都要求显式传入 enum，不接受裸字符串。

`special_token_ids` 必须显式传入 name 到全局 token id 的 mapping；`modality_ranges` 必须显式传入
`ModalityRange`。layout 不自动分配 special token id，也不自动推导 modality 起点。

special token id 可以数值上落在某个 modality range 里，但这个位置会被视为 reserved hole：
`to_global()` / `to_local()` 不会把它当作普通 modality token，head view 也会从普通
modality block 中跳过它。

## Tokenizer 组合

`MultiTokenizer` 持有多个实现 `SubTokenizer` protocol 的对象。tokenizer 只处理自己的
local ids，不知道也不依赖 global layout；下游在组模型输入时再用 `TokenLayout` 显式做
local/global id 映射。HF tokenizer 这类已有 `encode()` / `decode()` 的对象可以直接作为
mapping value 传入。

```python
from anytrain.idspace import Modality, ModalityRange, MultiTokenizer, TokenLayout

layout = TokenLayout(
    {"bos": 0, "eos": 1},
    [ModalityRange(Modality.TEXT, 2, 32000), ModalityRange(Modality.AUDIO, 32002, 1024)],
)
tokenizer = MultiTokenizer({
    Modality.TEXT: text_tokenizer,
    Modality.AUDIO: audio_tokenizer,
})

local_ids = tokenizer.encode(Modality.TEXT, text)
input_ids = [
    layout.special_token_id("bos"),
    *layout.to_global(Modality.TEXT, local_ids),
    layout.special_token_id("eos"),
]
text = tokenizer.decode(Modality.TEXT, local_ids)
```

规则：

- `encode()` 返回 modality-local ids。
- `decode()` 接收 modality-local ids。
- shared special token 插入和 local/global id 转换由下游显式调用 `TokenLayout` 完成。
- mixed sequence decode 不自动猜模态；下游需要保留 span 或显式按 modality decode。

## HF Tokenizer 迁移

`HFTokenizerAdapter` 用于从 Hugging Face 风格 tokenizer 拆出 special 和 regular text token：

```python
from anytrain.idspace import HFTokenizerAdapter, Modality

adapter = HFTokenizerAdapter.from_tokenizer(hf_tokenizer, modality=Modality.TEXT)
layout = adapter.layout
```

它会把 HF special token 作为 sparse shared special token id 记录在 `layout.special_token_ids` 中；text
modality 的 local/global id 仍然保留原 HF token id，不再把 regular vocab 压成 compact
block，也不再迁移 special token id。这样迁移时不需要理解 tokenizer 内部 id 语义，只会在 text
embedding 里浪费少量 special token 对应的位置。

## Embedding

`TokenEmbedding` 消费 `TokenLayout`。special token 使用 `nn.ParameterDict` 按名字保存少量
独立向量；每个模态使用自己的 `nn.Embedding`：

```python
from anytrain.idspace import Modality, TokenEmbedding

embed = TokenEmbedding(layout, dim)
h = embed(input_ids)
```

如果传入了显式 `special_embeddings` 或 `modality_embeddings`，`dim` 可以传 `None`，
此时会从第一组显式权重推断；完全默认初始化时必须显式传入 `dim`。显式权重可以只覆盖
一部分 special token 或 modality，缺失部分会按同一个 `dim` 默认初始化；未知 special 名字或
未知 modality 会报错。

special token 的初始化来源由调用方决定。比如同一个 `eos` 在 text/audio 预训练 embedding
里都有候选行，调用方可以先从任意一侧拷贝出 `nn.Parameter` 放进 `special_embeddings`；
`TokenEmbedding` 只保存最终参数，不记录这个向量来自哪个 modality。

`TokenEmbedding.as_head()` 返回的私有 head view 和 `TokenEmbedding` 共享同一批权重；它是 compact view，输出列从 0 开始，
不再等于 global vocab id：

```python
head = embed.as_head()
logits = head(h)
head_ids = head.to_head_ids(global_ids)
global_ids = head.to_global_ids(head_ids)

audio_head = embed.as_head(modalities=[Modality.AUDIO])
audio_only_head = embed.as_head(special_tokens=False, modalities=[Modality.AUDIO])
```

训练期不创建单独的大 `lm_head.weight`。head view 不额外注册参数；它按选定的 special token
和 modality span 写入 logits，并保留 head-local 到 global id 的双向映射。head view 只能由
`TokenEmbedding.as_head()` 创建，不作为公开类导出。下游如果要算
loss，记得先把 global labels 映射到 head ids。

## 扩展 HF Qwen3

给已有 Qwen3 tokenizer/model 加 audio modality 时，推荐保留 HF tokenizer 作为 text
sub-tokenizer，再额外追加 audio block：

```python
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

from anytrain.idspace import (
    HFTokenizerAdapter,
    Modality,
    ModalityRange,
    MultiTokenizer,
    TokenEmbedding,
    TokenLayout,
)

name = "Qwen/Qwen3-0.6B"
hf_tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name)

text = HFTokenizerAdapter.from_tokenizer(hf_tokenizer, modality=Modality.TEXT)
old_embed = model.get_input_embeddings()
text_vocab = max(text.layout.modality_range(Modality.TEXT).vocab_size, old_embed.num_embeddings)
audio_vocab = 1024

layout = TokenLayout(
    text.layout.special_token_ids,
    [
        ModalityRange(Modality.TEXT, 0, text_vocab),
        ModalityRange(Modality.AUDIO, text_vocab, audio_vocab),
    ],
)
tokenizer = MultiTokenizer({
    Modality.TEXT: text,
    Modality.AUDIO: audio_tokenizer,
})

embed = TokenEmbedding(
    layout,
    old_embed.embedding_dim,
    device=old_embed.weight.device,
    dtype=old_embed.weight.dtype,
)
with torch.no_grad():
    for name, token_id in text.layout.special_token_ids.items():
        embed.special_embeddings[name].copy_(old_embed.weight[token_id])
    embed.modality_embeddings[Modality.TEXT].weight[: old_embed.num_embeddings].copy_(
        old_embed.weight
    )

model.set_input_embeddings(embed)
model.config.vocab_size = layout.vocab_size
```

Qwen3 默认通常是 untied `lm_head`。这种情况下不要静默 tie；要么自己在训练环里把 label
映射到 head-local ids，要么继续保留独立输出头，把 compact head 当成显式投影层用。

## 更新策略

`TokenEmbedding` 不内置 optimizer policy。下游可以直接按模块分组：

```python
optimizer = torch.optim.AdamW(
    [
        {"params": embed.special_embeddings.parameters(), "lr": 1e-5},
        {"params": embed.modality_embeddings[Modality.TEXT].parameters(), "lr": 0.0},
        {"params": embed.modality_embeddings[Modality.AUDIO].parameters(), "lr": 1e-4},
    ]
)
```

## 非目标

当前不做：

- 不训练 tokenizer。
- 不定义多模态 batch schema。
- 不自动 interleave 不同模态序列。
- 不自动修改 Hugging Face model config 之外的完整模型结构。
- 不静默处理未知 id。
- 不支持不同 block 的 embedding dim 不一致。
