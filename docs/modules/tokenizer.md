# `anytrain.tokenizer` Design

## 定位

`anytrain.tokenizer` 放 tokenizer 算法组件。统一 token id space 和 token embedding/head
已独立到 [`anytrain.idspace`](idspace.md)。

当前公开对象：

- `CodecBPE`
- `EvalStats`

## CodecBPE

`CodecBPE` 用于在离散 codec frame sequence 上训练和评估 BPE。训练和语料入口只接受
2D 语义的 frame 序列：单 codebook 也写作 `[[id], ...]`，多 codebook 写作
`[[codebook_0_id, codebook_1_id, ...], ...]`。它内部把 frame 按 `codebook_sizes`
mixed-radix 编码成统一的 int sequence，再训练 compact BPE vocab，并持有
`tokenizers.models.BPE` 执行分词。

它支持：

- `train(corpus, codebook_sizes=..., ..., show_progress=True)`
- `encode(frames)`
- `decode(token_ids)`
- `repeat_interleave(x, token_ids, dim=...)`
- `eval(corpus, show_progress=True)`
- `save_pretrained(path)` / `from_pretrained(path)`

`path` 是 artifact 目录；`save_pretrained(path)` 只写出 `codec_bpe.json`，
`from_pretrained(path)` 只从该目录读取 `codec_bpe.json`。

基础用法：

```python
from anytrain.tokenizer import CodecBPE

def corpus():
    return [[[1], [2], [1], [2], [3]], [[1], [2], [3]]]

bpe = CodecBPE.train(
    corpus,
    codebook_sizes=(8192,),
    vocab_size=8,
)
token_ids = bpe.encode([[1], [2], [1], [2], [3]])
frames = bpe.decode(token_ids)
stats = bpe.eval([[[1], [2], [1], [2], [3]]], show_progress=False)
compression_ratio = stats.compression_ratio
vocab_size = bpe.vocab_size
```

多 codebook codec frame 用同一个入口：

```python
def multi_codebook_corpus():
    return [[[1, 4], [2, 7], [1, 4], [2, 7]]]

bpe = CodecBPE.train(
    multi_codebook_corpus,
    codebook_sizes=(4, 16),
    vocab_size=8,
)
token_ids = bpe.encode([[1, 4], [2, 7]])
frames = bpe.decode(token_ids)
```

`encode(frames)` / `decode(token_ids)` 只承诺 Python sequence 输入输出。空序列、
未知 frame / token id 一律报错。

`repeat_interleave` 用于把 BPE token 级张量展开到原始 frame 粒度：

- `token_ids` 必须是 1D，返回 `expanded_x, expanded_frames`。
- `expanded_frames` 的末维总是 codebook 维度，例如 `[T, 1]` 或 `[T, num_codebooks]`。

`CodecBPE` 的 BPE token id 是内部 compact vocab index。原始 codec frame
只通过 `decode()` 还原，不保证和 BPE token id 相同。
encode 唯一走 `tokenizers.models.BPE.tokenize`。训练入口使用
`tokenizers.trainers.BpeTrainer`。当前暴露的训练参数按 `BpeTrainer` 命名和含义对齐：
`vocab_size`、`min_frequency`、`show_progress`、`max_token_length`。`vocab_size` 包含
alphabet，因此如果 corpus 中不同 frame 数量更多，最终 compact vocab 会保留完整 alphabet
并可能大于传入值。`special_tokens`、`limit_alphabet` 和 `initial_alphabet` 这类文本
tokenizer 参数暂不暴露，因为 `CodecBPE` 的每个 token 都必须能无损还原成 codec frame。
训练会两遍扫描 corpus：第一遍收集完整 observed alphabet 并校验 frame，第二遍交给
`BpeTrainer` 学习 merges。传入的 corpus 必须是 callable，并在每次调用时返回可重新遍历的
frame sequence iterable。`show_progress=True` 时第一遍 alphabet scan 和第二遍 BPE
trainer 各自显示独立进度。
`eval()` 也支持 `show_progress`，用于评估大语料压缩率和 BPE token 使用分布时显示独立进度。
它返回扁平的 `EvalStats`：压缩指标里 `original_frames` 是 codec frame 数，
`encoded_tokens` 是 BPE token 数；同时包含 token 出现次数直方图、top-k token、实际使用
token 数、vocab 覆盖率、entropy，以及 vocab / eval 使用视角的 token 展开长度分布。
`top_token_counts` 每项是 `(token_id, count, frequency, length)`。长度分布是 dense
tuple，下标就是 BPE token 展开后的原始 frame 数，例如 `used_token_length_counts[3]`
表示长度为 3 的 BPE token 在 eval corpus 中出现了多少次；tuple 的最大下标就是当前统计中
最大的 BPE 展开长度。`eval()` 不返回完整 token-id 到 count 或 length 的大 dict；具体
bpe id 的频率明细只保留 `top_token_counts`，默认 `top_k=100`。
长度定义为一个 BPE token 通过 `decode()` 还原后覆盖的原始 codec frame 数。
Python sequence 入口中，frame 长度必须等于 `len(codebook_sizes)`，每个 code id
必须在对应 book size 范围内。单 codebook 也必须通过 `[id]` 表达，不保留 1D
unit sequence 入口；1D tensor 入口仅作为无 batch 便利接口。

`tokenizers` 不进入 package root import 链；只有使用 `CodecBPE` 需要构造底层 BPE model 时才要求安装。

## 边界

`tokenizer` 不做：

- 不定义 shared special token 或多模态 global id space。
- 不管理 embedding/head。
- 不解释多模态 batch schema。
- 不自动处理下游模型的 padding side、label mask 或 loss。
