# `anytrain.module` Design

## 定位

`anytrain.module` 放 task-agnostic 的 `torch.nn.Module` 积木，用于下游普通 LightningModule 显式组合。

它不是模型 zoo，也不提供完整任务网络。组件必须保持小而清楚，只依赖 core torch、einops 或对应 optional extra。

## 当前状态

当前提供这些跨项目复用组件：

- `ADTConfig`
- `AdaptiveDirichletTempering`
- `ADT` 兼容别名
- `DynamicConv1d`
- `DynamicConv2d`
- `DynamicConvTranspose1d`
- `ADTRouter1d`
- `ADTRouter2d`
- `MultiScalePool1d`
- `make_qwen3_config`
- `build_qwen3_rms_norm`
- `build_qwen3_rotary_embedding`
- `build_qwen3_attention`
- `build_qwen3_mlp`
- `build_qwen3_decoder_layer`
- `build_qwen3_model`
- `require_qwen3_class`
- `FSQConfig`
- `FiniteScalarQuantizer`
- `VQConfig`
- `EmbeddingVectorQuantizer`
- `GVQConfig`
- `GroupedVectorQuantizer`
- `RVQConfig`
- `ResidualVectorQuantizer`
- `QuantizeOutput`
- `QuantizationLoss`

ADT 用于 MoE/router logits 的自适应温度缩放和专家利用率统计，适合跨 audio、vision、text 等项目复用。

Dynamic Conv 用于按样本或按分段动态组合 expert convolution kernel。它保持 task-agnostic：

- 核心卷积只接收外部 `router`，或通过 `forward_manually(x, expert_weights)` 显式传入权重。
- 默认 `ADTRouter1d` 只依赖 torch 原生层和 `AdaptiveDirichletTempering`。
- 当前提供 1D conv / transposed conv，以及用于 Conv2d subsampling 的最小
  `DynamicConv2d`；2D transpose、segment/chunk 和 causal padding 等能力等真实下游
  需求明确后再补。
- router 输出被视作 expert 权重，不在 `DynamicConv1d` 内猜测 logits/softmax 语义。

`anytrain.module.qwen3` 是 Hugging Face Qwen3 的薄复用层，不重新实现
transformer block，也不要求下游维护一份庞大的项目级 config：

- `anytrain.module` 顶层只导出 builder 和 resolver，所以默认 import 不依赖
  `transformers`。
- `anytrain.module.qwen3.Qwen3RMSNorm`、`Qwen3MLP`、`Qwen3Attention` 等名字通过
  module `__getattr__` 按需解析到
  `transformers.models.qwen3.modeling_qwen3`。
- `build_qwen3_mlp()`、`build_qwen3_attention()`、`build_qwen3_decoder_layer()` 等
  builder 只暴露对应模块初始化需要的少量参数，内部临时生成 Hugging Face
  `Qwen3Config`。
- 如果下游确实要构建完整 HF model，可以显式调用 `make_qwen3_config()` 或
  `build_qwen3_model()`；普通 codec 组合应优先依赖更窄的 builder。
- 缺少 `transformers` 或版本不含 Qwen3 时，builder 会抛出明确的 `ImportError`。

Quantization 提供 task-agnostic 的 FSQ、embedding-table VQ、GVQ 和 RVQ：

- `FiniteScalarQuantizer` 使用 `levels` 描述每个 scalar code dimension 的 level 数；它和普通 VQ 一样是单 codebook 接口，`indices` shape 为 `(...)`，`codebook_vectors` shape 为 `(..., codebook_dim)`。
- FSQ 默认使用 odd-only `levels` preset，推荐 odd `levels` 以保持 scalar grid 关于 0 对称；even `levels` 会触发 warning，但仍使用保留 0 码点的 zero-friendly grid。
- FSQ 的 `bound_scale` 控制进入 `tanh` 前的 latent 尺度；调大它可以缓解大幅值 projected latents 的边界饱和。
- `EmbeddingVectorQuantizer` 是 embedding table + nearest-neighbor VQ，实现名避免和 protocol 混淆。
- `GroupedVectorQuantizer` 是 learned product codebook，例如 `group_sizes=(90, 90)` 对外表现为 `codebook_size=8100` 的单 codebook，但内部只搜索两个 90-size group。
- `ResidualVectorQuantizer` 组合多个 embedding VQ，第一版要求统一 `input_dim`、`codebook_dim` 和 `codebook_size`；`latents_to_codebook_vectors()` 不触发 dropout、loss 或 EMA 更新。
- 量化输出统一为 `QuantizeOutput`，离散整数用 `indices`，连续 codebook 空间向量用 `codebook_vectors`。
- RVQ dropout 下 inactive codebook 使用 `indices == -1` 和 `active_codebook_mask == False` 标记，不能只靠 `codebook_vectors` 判断有效性。

迁移设计见 [`docs/quantization-migration.md`](../quantization-migration.md)。第一版不迁入 audio codec、model zoo 或 deepaudio 的 projector/MoE 体系。

多模态 token id space 和 block embedding 路由已独立到 `anytrain.idspace`，
设计见 [`docs/modules/idspace.md`](idspace.md)。

## 依赖策略

`torch` 和 `einops` 是默认依赖。`einops.rearrange` 用于让动态分组卷积的 batch/channel/kernel shape 变换保持可读。

`anytrain.module` 后续如有额外三方依赖，使用 `module` extra 管理：

```bash
pip install anytrain[module]
```

当前 Qwen3 复用层需要 `module` extra 中的 `transformers`。ADT、Dynamic Conv 和
quantization 不需要 optional 依赖；`import anytrain` 和 `import anytrain.module`
不会主动导入 `transformers`。
