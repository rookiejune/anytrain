# `anytrain.tokenizer` Design

## 定位

`anytrain.tokenizer` 放 tokenizer 算法组件。统一 token id space、HF tokenizer 迁移、
multi tokenizer 组合和 token embedding/head 已独立到 [`anytrain.idspace`](idspace.md)。

当前公开对象：

- `IntBPE`
- `Merge`
- `CompressionStats`

## IntBPE

`IntBPE` 用于在 int sequence 上训练和评估 BPE。它保持真实 int 语义，同时内部持有
`tokenizers.models.BPE`，用于和 Hugging Face `tokenizers.Tokenizer(model=...)` 对接。

它支持：

- `train(corpus, ...)`
- `encode_units(units)`
- `expand_ids(token_ids)`
- `expand_with_counts(token_ids)`
- `repeat_interleave(x, token_ids, dim=...)`
- `eval(corpus)`
- `save_pretrained(path)` / `from_pretrained(path)`

基础用法：

```python
from anytrain.tokenizer import IntBPE

bpe = IntBPE.train([[1, 2, 1, 2, 3], [1, 2, 3]], vocab_size=8)
token_ids = bpe.encode_units([1, 2, 1, 2, 3])
unit_ids = bpe.expand_ids(token_ids)
stats = bpe.eval([[1, 2, 1, 2, 3]])
```

`tokenizers` 不进入 package root import 链；只有使用 `IntBPE` 需要构造底层 BPE model 时才要求安装。

## 边界

`tokenizer` 不做：

- 不定义 shared special token 或多模态 global id layout。
- 不管理 embedding/head。
- 不解释多模态 batch schema。
- 不自动处理下游模型的 padding side、label mask 或 loss。
