# Tokenizer TODO

先做 BPE 的 base-unit 展开和对齐重复，不急着引入完整 tokenizer 框架。

## BPE Base Unit

目标：在 `tokenizers.models.BPE` 默认行为之外，增加两个可选能力：

1. 将每个 BPE token id 展开到它递归合并前的 base units。
2. 给定和 `token_ids` 在序列维度对齐的 `x`，按每个 token 展开的 base 数量做 `repeat_interleave`。

默认配置必须保持 Hugging Face `tokenizers` BPE 的行为不变；只有显式调用展开接口或打开 `expand_base` 时才改变输出。

### 术语

- `token`：父类 BPE 输出的 token，通常是已经按最长 merge 匹配后的 token。
- `unit`：不能再通过 BPE merge 反向拆分的最小 token。第一版把 vocab 中没有 merge children 的 token 视为 base。
- `token_id`：BPE vocab id，指向父类 BPE 输出的 token。
- `unit_id`：BPE vocab id，指向展开后的 unit。
- `special token`：默认作为 atomic unit，不参与递归拆分；如果用户希望拆分，需要显式传入配置。
- `token_ids`：和模型输入序列对齐的一维或批量 BPE vocab id 张量。避免命名为 `word_ids`，因为它容易和 `Encoding.word_ids()` 的原始词索引语义混淆。

### 接口草案

已验证 `tokenizers.models.BPE` 不能被 Python subclass：

```text
TypeError: type 'tokenizers.models.BPE' is not an acceptable base type
```

因此第一版不再公开旧 `BPE` 类；保留内部 `_CoreBPE` 负责 int sequence 训练和展开，公开入口命名为 `IntBPE`：

```python
class _CoreBPE:
    def unit_ids(self, token_id: int) -> tuple[int, ...]: ...
    def expand_ids(self, token_ids: Sequence[int]) -> list[int]: ...
    def expand_with_counts(self, token_ids: Sequence[int]) -> tuple[list[int], list[int]]: ...
    def repeat_interleave(
        self,
        x: torch.Tensor,
        token_ids: torch.Tensor,
        *,
        dim: int = -2,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


class IntBPE:
    def __init__(
        self,
        *,
        cache_capacity: int | None = None,
        dropout: float | None = None,
        unk_token: str | None = None,
        continuing_subword_prefix: str | None = None,
        end_of_word_suffix: str | None = None,
        fuse_unk: bool | None = None,
        byte_fallback: bool = False,
        ignore_merges: bool = False,
    ) -> None: ...

    @classmethod
    def train(cls, corpus: Iterable[Sequence[int]], ...) -> Self: ...

    @classmethod
    def from_dict(cls, state: IntBPEState, ...) -> Self: ...
    def from_pretrained(cls, path: str | Path, ...) -> Self: ...
    def save_pretrained(self, path: str | Path) -> Path: ...

    model: tokenizers.models.BPE
    core: _CoreBPE
```

`IntBPE.__init__()` 只接底层 `tokenizers.models.BPE` 的运行参数，不要求用户手写 `tokens` / `merges`。日常训练走 `IntBPE.train(...)`；内存状态走 `IntBPE.from_dict(...)`；磁盘 artifact 走 `save_pretrained(...)` / `from_pretrained(...)`。
`IntBPE` 负责 int sequence 训练、编码和展开；它内部持有 `tokenizers.models.BPE` 用于和 `Tokenizer(model=...)` 对接。
由于 `tokenizers.models.BPE` 从字符串字符级符号开始 tokenize，`IntBPE` 把每个原始 int unit 映射到一个私用区 Unicode 字符；`core` 仍然保存真实 int 语义。

### 展开规则

1. 初始化时从 `vocab` 和 `merges` 建立 `token_id -> tuple[unit_id, ...]` 的递归表。
2. 对 merge 产生的 token，递归展开左、右 children，直到 base unit。
3. 对 special token、unknown token、没有 merge 信息的 token：
   - `strict=True` 时直接抛错，避免训练数据被悄悄错位。
   - `strict=False` 时作为 atomic unit 返回自身。
4. `expand_ids([t0, t1])` 返回拼接后的 unit ids。
5. `expand_with_counts([t0, t1])` 同时返回 `unit_ids` 和每个 token 对应的展开长度，例如 `[2, 1]`。

### Int Sequence 训练行为

下游训练语料不是自然语言 string，而是一批 int sequence，例如 codec token、离散语音 token 或其他已经数值化的序列：

```python
corpus = [
    [1, 2, 1, 2, 3],
    [1, 2, 3],
]
```

第一版训练目标：

1. 直接在 `Sequence[Sequence[int]]` 上训练，不把 int 静默转成文本 token。
2. 初始 vocab 的每个 unit 保留原始 int 作为 base token id，例如 token id `1` 对应 unit `(1,)`，token id `2` 对应 unit `(2,)`。
3. 每次 merge 选择当前语料里最频繁的相邻 token pair，新增一个 token，它的 unit 展开是左右 token 展开的拼接。
4. 合并后的 token 可以表示多个原始 int，例如 `(1, 2)`、`(1, 2, 3)`。
5. `encode_units([1, 2, 1, 2, 3])` 使用训练得到的 merges 做最长可用合并，返回压缩后的 `token_ids`。
6. `expand_ids(token_ids)` 必须可以还原为原始 int 序列；这是第一版最重要的不变量。

命名约定：

- 原始语料里的 int 叫 `unit`。
- BPE 合并后的编号叫 `token_id`。
- `tokens` 存 `token_id -> tuple[int, ...]`，tuple 内容始终是原始 unit；merge token id 从 `max(unit) + 1` 开始递增。

测试行为：

1. 用 `[[1, 2, 1, 2, 3], [1, 2, 3]]` 训练，频繁 pair `(1, 2)` 应先被合并。
2. 在继续训练后，应能产生表示多个 int 的 token，例如 `(1, 2, 3)`。
3. `encode_units(seq)` 后再 `expand_ids(...)` 必须等于原始 `seq`。
4. 合并后的 token 展开长度决定 `repeat_interleave` 的重复次数。
5. 空语料、空序列、未知 unit 在 `strict=True` 下直接抛错。

### `repeat_interleave` 语义

输入约束：

- `token_ids` 在序列维度上和 `x` 对齐。
- 第一版先支持单条序列：`token_ids.shape == (seq,)`，`x.size(dim) == seq`。
- 批量序列会因为每条样本展开后长度不同而变成 ragged，第二版再提供 padding/mask 或 packed offsets。

单条序列行为：

```text
token_ids = [10, 20]
unit_ids(10) = [1, 2, 3]
unit_ids(20) = [4]

expanded_unit_ids = [1, 2, 3, 4]
expanded_x = [x[0], x[0], x[0], x[1]]
```

返回值：

- `expanded_x`：沿 `dim` 重复后的 tensor。
- `expanded_unit_ids`：展开后的 unit id tensor，和 `expanded_x` 的序列维度对齐。

实现时使用 `torch.repeat_interleave`，`counts` tensor 必须放在 `x.device`；返回 tensor 保持 `x` 的 dtype/device，展开后的 ids 保持 `token_ids` 的 dtype/device。

### 后续批量支持

批量输入单独设计，避免第一版把 ragged 行为写乱：

1. `repeat_interleave_packed(x, token_ids, dim=1)`：
   - 返回 concat 后的 `expanded_x`、`expanded_unit_ids`、`offsets`。
   - 适合后续接 pack/pad 或自定义 loss。
2. `repeat_interleave_padded(x, token_ids, dim=1, pad_id=0)`：
   - 返回 `padded_x`、`padded_unit_ids`、`mask`。
   - 每个 batch item pad 到本 batch 最大展开长度。

### 依赖边界

- `tokenizers` 不进 core import 链；如果当前环境没有安装，在使用 `anytrain.tokenizer.int_bpe` 时给清晰错误。
- `pyproject.toml` 后续把 `tokenizers` 放进 `text` extra，或者新增 `tokenizer` extra 后由 `text` 聚合。
- 不在 tokenizer 模块里解释 batch schema，也不自动处理下游模型的 padding side。

### 测试计划

1. 用小 vocab 和 merges 构造 `a + b -> ab`、`ab + c -> abc`，确认 `abc` 展开为 `[a, b, c]`。
2. 默认 tokenize 路径和父类 BPE 行为一致。
3. special token 默认 atomic；`strict=True` 遇到不可展开 token 抛错。
4. `expand_ids` 和 `expand_with_counts` 的顺序、长度、空输入行为正确。
5. `repeat_interleave` 对非法 shape 抛清楚错误。
6. `repeat_interleave` 保持 `x` dtype/device，并让 `expanded_unit_ids` 和展开后的序列长度一致。

### 实现步骤

1. Done: 先实现纯 int sequence BPE，保证训练、编码、展开、`repeat_interleave` 闭环。
2. Done: 确认 `tokenizers.models.BPE` 不能被 Python subclass；后续只能走 composition。
3. Done: 增加 `IntBPE`，持有 `core: _CoreBPE` 和 `model: tokenizers.models.BPE`，不改变 int 训练语义。
4. Done: 把 `IntBPE.__init__()` 改成只接底层 backend 运行参数；手动 state 构造改为 `from_dict(...)`，磁盘加载保存使用 `from_pretrained(...)` / `save_pretrained(...)`。
5. Done: 删除公开 `BPE` 类名，把纯 int core 收成内部 `_CoreBPE`。
6. Done: 增加 `eval(corpus)`，在 train 后直接统计原始长度、压缩后长度、ratio、factor 和 gain。
7. Done: 增加测试覆盖训练 merge、未知 unit、空输入、repeat 对齐和压缩统计。
8. Later: 增加 batch packed/padded 两种输出形态。
9. Later: 根据下游实际调用方式决定是否让 `Tokenizer.encode()` 直接返回 base units，或只保留显式 post-process helper。

## Idspace Compose

Done: special-aware 主线已迁到 `anytrain.idspace`：

- `anytrain.idspace.layout` 提供 `TokenLayout`、`Modality` 和 `ModalityRange`，让 shared special token 独立于写死的 modality block。
- `anytrain.idspace.tokenizer` 提供 `MultiTokenizer` / `SubTokenizer`，只处理各模态 local id。
- `anytrain.idspace.hf` 提供 `HFTokenizerAdapter`，从 Hugging Face 风格 tokenizer 中拆出 sparse special id 和 regular text range。
- adapter 不迁移 token id；text range 保留原 HF token id 范围，避免 regular token 再映射。
- `anytrain.tokenizer` 只保留 tokenizer 算法组件，例如 `IntBPE`。
