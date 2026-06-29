# `anytrain.tokenizer` Design

## 定位

`anytrain.tokenizer` 放 tokenizer 算法组件。统一 token id space 和 token embedding/head
已独立到 [`anytrain.idspace`](idspace.md)。

当前公开对象：

- `CodecBPE`
- `Merge`
- `CompressionStats`

## CodecBPE

`CodecBPE` 用于在离散 unit sequence 上训练和评估 BPE。unit 可以是 `int`，
也可以是固定长度的 `Sequence[int]`，用于多 codebook codec frame。它内部持有
`tokenizers.models.BPE`，用于和 Hugging Face `tokenizers.Tokenizer(model=...)` 对接。

它支持：

- `train(corpus, ..., progress=False)`
- `encode_units(units)`
- `expand_ids(token_ids)`
- `expand_with_counts(token_ids)`
- `repeat_interleave(x, token_ids, mask=None, dim=...)`
- `eval(corpus)`
- `save_pretrained(path)` / `from_pretrained(path)`

`save_pretrained(path)` 会写出 `codec_bpe.json` 和 `tokenizer.json`。

基础用法：

```python
from anytrain.tokenizer import CodecBPE

bpe = CodecBPE.train([[1, 2, 1, 2, 3], [1, 2, 3]], vocab_size=8)
token_ids = bpe.encode_units([1, 2, 1, 2, 3])
unit_ids = bpe.expand_ids(token_ids)
stats = bpe.eval([[1, 2, 1, 2, 3]])
vocab_size = bpe.vocab_size
```

多 codebook codec frame 用同一个入口：

```python
bpe = CodecBPE.train([[(1, 4), (2, 7), (1, 4), (2, 7)]], vocab_size=8)
token_ids = bpe.encode_units([(1, 4), (2, 7)])
frames = bpe.expand_ids(token_ids)
```

`repeat_interleave` 用于把 BPE token 级张量展开到原始 unit 粒度：

- `token_ids` 是 1D 时保持简单行为，返回 `expanded_x, expanded_unit_ids`。
- `token_ids` 是 2D 时返回 batch padded 结果：
  `expanded_x, expanded_unit_ids, expanded_mask`。
- 2D 输入可传 `mask` 标记有效 token。展开时只使用有效 token，输出按 batch 内最长展开长度重新 pad。
- 输出 `expanded_unit_ids` 的 padding id 从 `input_ids[~mask]` 推断；若 padding 位存在多个不同值会报错。若需要输出 padding 但无法推断 padding id，也会报错。
- 对 tuple unit，`expanded_unit_ids` 的末维是 unit 维度，例如 `[T, num_codebooks]`
  或 `[B, T, num_codebooks]`。

`CodecBPE` 的 BPE token id 是内部 compact vocab index。原始 unit id 或 codec frame
只通过 `expand_ids()` / `expand_with_counts()` 还原，不保证和 BPE token id 相同。
训练时传 `progress=True` 会显示 corpus 读取和 merge 轮次两个 `tqdm` 进度条。
若传入 `vocab_size`，它按 compact BPE vocab 大小解释，必须不小于 corpus 中不同
unit 的数量。tuple unit 必须在同一个 tokenizer 内保持固定长度，不能和 int unit 混用。

`tokenizers` 不进入 package root import 链；只有使用 `CodecBPE` 需要构造底层 BPE model 时才要求安装。

## 边界

`tokenizer` 不做：

- 不定义 shared special token 或多模态 global id space。
- 不管理 embedding/head。
- 不解释多模态 batch schema。
- 不自动处理下游模型的 padding side、label mask 或 loss。
