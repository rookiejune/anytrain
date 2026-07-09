# `anytrain.idspace` Design

## 定位

`anytrain.idspace` 管 unified token id space：同一套 `IdSpace` 驱动多模态
embedding 路由和 tied head view。它不训练 tokenizer，不组合多个 tokenizer，不迁移已有 tokenizer，
不解释 batch schema，也不替下游修改完整模型。

当前公开对象：

- `Modality`
- `IdSpace`
- `ModalityBlock`
- `IdSpaceEmbedding`

## IdSpace

`IdSpace` 是下游 tokenizer 输出和 embedding 共享的 id 空间定义。普通 token 的 global id
在不同 modality 之间不能重叠；special token 是全局共享的 sparse token id set，不属于
任何 modality。embedding 先匹配 special token id，剩下的 id 再按 modality block 路由：

```text
special_token_ids: {pad: 0, bos: 151643, eos: 151645}
text:    [0, text_vocab)
audio:   [text_vocab, text_vocab + audio_vocab)
```

`Modality` 是写死的 `StrEnum`，第一版只包含 `Modality.TEXT` 和 `Modality.AUDIO`。
space 和 embedding 构造都要求显式传入 enum，不接受裸字符串。

`special_token_ids` 必须显式传入 name 到全局 token id 的 mapping；`modality_blocks` 必须显式传入
`ModalityBlock`。space 不自动分配 special token id，也不自动推导 modality 起点。

special token id 可以数值上落在某个 modality block 里，但这个位置会被视为 reserved hole：
`to_global()` / `to_local()` 不会把它当作普通 modality token，head view 也会从普通
modality block 中跳过它。

`regular_blocks(modality)` 返回某个 modality 里去掉 special holes 后的 global contiguous
blocks。`IdSpaceEmbedding.head_view()` 用它显式选择 special token 和 modality block，并维护
head-local/global id 映射。

`block_containing_id(token_id)` 只回答某个 global id 数值上落在哪个 modality block 内，
即使这个 id 同时是 special token；需要普通 token 语义时仍使用 `modality_block_for_id()` /
`to_local()`，它们会明确拒绝 special token。

## Embedding

`IdSpaceEmbedding` 消费 `IdSpace`。special token 使用 `nn.ParameterDict` 按名字保存少量
独立向量；每个模态使用自己的 `nn.Embedding`：

```python
from anytrain.idspace import IdSpace, IdSpaceEmbedding, Modality, ModalityBlock

space = IdSpace(
    {"bos": 0, "eos": 1},
    [ModalityBlock(Modality.TEXT, 2, 32000), ModalityBlock(Modality.AUDIO, 32002, 1024)],
)
embed = IdSpaceEmbedding(space, dim)
input_ids = embed.space.to_global(Modality.TEXT, local_ids)
h = embed(input_ids)
head = embed.head_view()
logits = head(h)
```

`dim` 是统一输出维。需要默认新建 embedding 时必须传入；若已有显式
`special_embeddings` / 未适配的 `modality_embeddings`，可从权重形状推断。若所有显式
modality embedding 都带 adapter，也必须显式传入 `dim`。构造器不接受 `device` / `dtype`，
新建参数用 PyTorch 默认值，之后由调用方 `.to(...)`。

special token 的向量解析顺序：

1. 显式 `special_embeddings[name]` 优先，用来注册独立向量或覆盖 modality 行。
2. 未注册但 id 落在某个 modality block 内时，直接复用该 modality 对应 local 行。
3. 未注册且不在任何 modality 内时，才默认随机初始化一个 `nn.Parameter`。

显式 `modality_embeddings` 可以只覆盖一部分 modality，缺失 modality 会按同一个 `dim`
默认初始化；未知 special 名字或未知 modality 会报错。显式 modality embedding 必须是
`nn.Module`，并暴露 `num_embeddings`、`embedding_dim`、`weight` 和可调用的 forward 行为。

可选 `adapters: Mapping[Modality, nn.Module]` 只作用于 `forward()`：lookup 原生行后再投影到
统一 `dim`，不会物化整张投影表。带 adapter 的 modality 允许原生 `embedding_dim != dim`；
没有 adapter 的 modality 仍要求 `embedding_dim == dim`。special token 始终落在统一 `dim`。

```python
embed = IdSpaceEmbedding(
    space,
    dim,
    modality_embeddings={
        Modality.TEXT: text_emb,   # embedding_dim == dim
        Modality.AUDIO: audio_emb, # embedding_dim may differ
    },
    adapters={Modality.AUDIO: nn.Linear(audio_emb.embedding_dim, dim, bias=False)},
)
h = embed(input_ids)  # audio rows go through adapter
```

`IdSpaceEmbedding` 对外尽量贴近 `nn.Embedding`：`forward()` 接收 global ids，
`num_embeddings` 等于 `space.vocab_size`，`embedding_dim` / `dim` 等于统一输出维。
`weight` 返回按 global id 拼出的 dense **原生**表，且要求所有 modality 原生维等于 `dim`；
存在变长原生表时访问 `weight` 会报错，应直接读
`modality_embeddings[modality].weight`。`weight` 不是单个 `nn.Parameter`；
optimizer 分组和 partial freeze 仍然通过 `special_embeddings` / `modality_embeddings` /
`adapters` 这些真实参数完成。

local/global id 转换仍然属于 `IdSpace`，推荐显式通过 `embed.space.to_global(...)` /
`embed.space.to_local(...)` 调用。`IdSpaceEmbedding` 不再转发这些方法，避免把 id 规则、
batch/span 解释和 embedding 参数混在同一个类里。

special token 的初始化来源由调用方决定。比如同一个 `eos` 在 text/audio 预训练 embedding
里都有候选行，调用方可以先从任意一侧拷贝出 `nn.Parameter` 放进 `special_embeddings`；
`IdSpaceEmbedding` 只保存最终参数，不记录这个向量来自哪个 modality。

`IdSpaceEmbedding.head_view()` 返回一个 compact view，输出列从 0 开始，不再等于 global
vocab id：

```python
head = embed.head_view(special_tokens=["bos", "eos"], modalities=[Modality.AUDIO])
logits = head(h_audio_native)  # h must match native audio embedding dim
head_ids = head.to_head_ids(global_ids)
global_ids = head.to_global_ids(head_ids)
```

head view 不额外注册参数；它只保存选择出的 special token 和 modality span，并在 forward
时读取对应 embedding-like module 的当前 **原生** `weight`，因此它是 tied view，不经过
`adapters`。special 的解析规则与 input embedding 相同：显式 Parameter 优先，否则从所在
modality 的 `weight` 取行。所选 special / modality 的原生维必须一致，否则构造时报错；
不同原生维的 modality 应拆成多个 head，并由调用方自行把 hidden 投到对应空间。

## 更新策略

`IdSpaceEmbedding` 不内置 optimizer policy。下游可以直接按模块分组：

```python
optimizer = torch.optim.AdamW(
    [
        {"params": embed.special_embeddings.parameters(), "lr": 1e-5},
        {"params": embed.modality_embeddings[Modality.TEXT].parameters(), "lr": 0.0},
        {"params": embed.modality_embeddings[Modality.AUDIO].parameters(), "lr": 1e-4},
        {"params": embed.adapters.parameters(), "lr": 1e-4},
    ]
)
```

## 非目标

当前不做：

- 不训练 tokenizer。
- 不读取或迁移已有 tokenizer/model vocab。
- 不定义多模态 batch schema。
- 不自动 interleave 不同模态序列。
- 不组合多个 tokenizer。
- 不自动修改下游模型结构或 config。
- 不静默处理未知 id。
- 不在 tied head / `weight` 路径上自动投影变长 embedding；`adapters` 只服务 input `forward()`。
