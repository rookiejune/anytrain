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
logits = h @ embed.weight.T
```

如果传入了显式 `special_embeddings` 或 `modality_embeddings`，`dim` 可以传 `None`，
此时会从第一组显式权重推断；完全默认初始化时必须显式传入 `dim`。显式权重可以只覆盖
一部分 special token 或 modality，缺失部分会按同一个 `dim` 默认初始化；未知 special 名字或
未知 modality 会报错。

`IdSpaceEmbedding` 对外尽量贴近 `nn.Embedding`：`forward()` 接收 global ids，
`num_embeddings` 等于 `space.vocab_size`，`embedding_dim` 等于内部向量维度，
`weight` 返回按 global id 拼出的 dense tensor。`weight` 不是单个 `nn.Parameter`；
optimizer 分组和 partial freeze 仍然通过 `special_embeddings` / `modality_embeddings`
这些真实参数完成。

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
logits = head(h)
head_ids = head.to_head_ids(global_ids)
global_ids = head.to_global_ids(head_ids)
```

head view 不额外注册参数；它只保存选择出的 special token 和 modality span，并在 forward
时读取对应 embedding-like module 的当前 `weight`，因此它是 tied view。下游如果需要让输入侧
audio embedding 先经过投影，应把投影逻辑放进被选中的 embedding-like module 的 `weight`
里；如果 hidden 空间不同，再在 head 前显式接 output adapter。

## 更新策略

`IdSpaceEmbedding` 不内置 optimizer policy。下游可以直接按模块分组：

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
- 不读取或迁移已有 tokenizer/model vocab。
- 不定义多模态 batch schema。
- 不自动 interleave 不同模态序列。
- 不组合多个 tokenizer。
- 不自动修改下游模型结构或 config。
- 不静默处理未知 id。
- 不支持不同 block 的 embedding dim 不一致。
