# `anytrain.tokenizer` Design

## 定位

`anytrain.tokenizer` 放 tokenizer 算法组件。统一 token id space 和 token embedding/head
已独立到 [`anytrain.idspace`](idspace.md)。

当前公开对象：

- `CodecBPE`
- `Merge`
- `CompressionStats`

## CodecBPE

`CodecBPE` 用于在离散 codec frame sequence 上训练和评估 BPE。公开入口只接受
2D 语义的 frame 序列：单 codebook 也写作 `[[id], ...]`，多 codebook 写作
`[[codebook_0_id, codebook_1_id, ...], ...]`。它内部把 frame 按 `codebook_sizes`
mixed-radix 编码成统一的 int sequence，再训练 compact BPE vocab，并持有
`tokenizers.models.BPE` 用于和 Hugging Face `tokenizers.Tokenizer(model=...)` 对接。

它支持：

- `train(corpus, codebook_sizes=..., ..., show_progress=True)`
- `encode_frames(frames)`
- `expand_ids(token_ids)`
- `expand_with_counts(token_ids)`
- `repeat_interleave(x, token_ids, mask=None, dim=...)`
- `eval(corpus, show_progress=True)`
- `save_pretrained(path)` / `from_pretrained(path)`

`save_pretrained(path)` 会写出 `codec_bpe.json` 和 `tokenizer.json`。

基础用法：

```python
from anytrain.tokenizer import CodecBPE

bpe = CodecBPE.train(
    [[[1], [2], [1], [2], [3]], [[1], [2], [3]]],
    codebook_sizes=(8192,),
    vocab_size=8,
)
token_ids = bpe.encode_frames([[1], [2], [1], [2], [3]])
frames = bpe.expand_ids(token_ids)
stats = bpe.eval([[[1], [2], [1], [2], [3]]], show_progress=False)
vocab_size = bpe.vocab_size
```

多 codebook codec frame 用同一个入口：

```python
bpe = CodecBPE.train(
    [[[1, 4], [2, 7], [1, 4], [2, 7]]],
    codebook_sizes=(4, 16),
    vocab_size=8,
)
token_ids = bpe.encode_frames([[1, 4], [2, 7]])
frames = bpe.expand_ids(token_ids)
```

`repeat_interleave` 用于把 BPE token 级张量展开到原始 frame 粒度：

- `token_ids` 是 1D 时保持简单行为，返回 `expanded_x, expanded_frames`。
- `token_ids` 是 2D 时返回 batch padded 结果：
  `expanded_x, expanded_frames, expanded_mask`。
- 2D 输入可传 `mask` 标记有效 token。展开时只使用有效 token，输出按 batch 内最长展开长度重新 pad。
- 输出 `expanded_frames` 的 padding id 从 `input_ids[~mask]` 推断；若 padding 位存在多个不同值会报错。若需要输出 padding 但无法推断 padding id，也会报错。
- `expanded_frames` 的末维总是 codebook 维度，例如 `[T, 1]`、`[B, T, 1]`
  或 `[B, T, num_codebooks]`。

`CodecBPE` 的 BPE token id 是内部 compact vocab index。原始 codec frame
只通过 `expand_ids()` / `expand_with_counts()` 还原，不保证和 BPE token id 相同。
训练入口使用 `tokenizers.trainers.BpeTrainer`。当前暴露的训练参数按
`BpeTrainer` 命名和含义对齐：`vocab_size`、`min_frequency`、
`show_progress`、`max_token_length`。`vocab_size` 包含 alphabet，因此如果
corpus 中不同 frame 数量更多，最终 compact vocab 会保留完整 alphabet 并可能大于
传入值。`special_tokens`、`limit_alphabet` 和 `initial_alphabet` 这类文本 tokenizer
参数暂不暴露，因为 `CodecBPE` 的每个 token 都必须能无损还原成 codec frame。
训练会两遍扫描 corpus：第一遍收集完整 observed alphabet 并校验 frame，第二遍交给
`BpeTrainer` 学习 merges。传入的 corpus 必须是可重放 iterable，或 callable 且每次返回
新的 iterator。`show_progress=True` 时第一遍 alphabet scan 和第二遍 BPE trainer
各自显示独立进度。
`eval()` 也支持 `show_progress`，用于评估大语料压缩率时显示独立进度。
frame 长度必须等于 `len(codebook_sizes)`，每个 code id 必须在对应 book size 范围内。
单 codebook 也必须通过 `[id]` 表达，不保留 1D unit 入口。

`tokenizers` 不进入 package root import 链；只有使用 `CodecBPE` 需要构造底层 BPE model 时才要求安装。

## 边界

`tokenizer` 不做：

- 不定义 shared special token 或多模态 global id space。
- 不管理 embedding/head。
- 不解释多模态 batch schema。
- 不自动处理下游模型的 padding side、label mask 或 loss。
