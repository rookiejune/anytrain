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
- `DEFAULT_FSQ_LEVELS`
- `default_fsq_levels`
- `FSQConfig`
- `FiniteScalarQuantizer`
- `VQConfig`
- `EmbeddingVectorQuantizer`
- `GVQConfig`
- `GroupedVectorQuantizer`
- `RVQConfig`
- `ResidualVectorQuantizer`
- `QuantizerProtocol`
- `QuantizerType`
- `QuantizeOutput`
- `QuantizationLoss`

`anytrain.module.dynamic_conv` 还公开 `eca_kernel_size`，用于按 channel 数计算 router 的 odd
kernel size。

`anytrain.module.quantization` 还公开：

- `AGRVQConfig`
- `AutoGroupResidualVectorQuantizer`

ADT 用于 MoE/router logits 的自适应温度缩放和专家利用率统计，适合跨 audio、vision、text 等项目复用。
统计收集路径会先把 FP16/BF16 logits 升到 FP32，再计算用于统计的 softmax、计数和各阶矩，
避免长序列 reduction 溢出；启用 distributed stats 同步时也使用 FP32 collective。forward
输出仍以调用方传入的 logits dtype 计算温度 softmax。

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

Quantization 提供 task-agnostic 的 FSQ、embedding-table VQ、GVQ、RVQ 和 AGRVQ：

- `FiniteScalarQuantizer` 使用 `levels` 描述每个 scalar code dimension 的 level 数；它和普通 VQ 一样是单 codebook 接口，`indices` shape 为 `(...)`，`codebook_vectors` shape 为 `(..., codebook_dim)`。
- FSQ 默认使用 odd-only `levels` preset，推荐 odd `levels` 以保持 scalar grid 关于 0 对称；even `levels` 会触发 warning，但仍使用保留 0 码点的 zero-friendly grid。
- FSQ 的 `bound_scale` 控制进入 `tanh` 前的 latent 尺度；调大它可以缓解大幅值 projected latents 的边界饱和。
- `EmbeddingVectorQuantizer` 是 embedding table + nearest-neighbor VQ，实现名避免和 protocol 混淆。
- nearest-neighbor 查找会沿 latent 和 codebook 两个维度自动分块，确保任一临时比较矩阵不
  超过内部元素上限，限制长序列和超大码本组合时的峰值显存。
- `EmbeddingVectorQuantizer` 配置 `VQConfig(use_ema=True)` 时使用 `bincount` / `index_add_`
  以 FP32 聚合并保存统计，即使 quantizer 整体转换为 FP16/BF16 也不降低 EMA 精度；
  `torch.distributed` 已初始化时会在更新 EMA 和 codebook 前全局求和 counts/sums。初始 EMA
  state/codebook 必须已在 rank 间同步，且所有 rank 必须以相同顺序参与相同 quantizer stage
  的 collective。当前 DDP 训练不能同时启用 RVQ 的 per-vector dropout 和 EMA；各 rank 独立
  采样可能让 collective 参与顺序不一致。
- `GroupedVectorQuantizer` 是 learned product codebook，例如 `group_sizes=(90, 90)` 对外表现为 `codebook_size=8100` 的单 codebook，但内部只搜索两个 90-size group。
- `ResidualVectorQuantizer` 组合多个 embedding VQ，第一版要求统一 `input_dim`、`codebook_dim` 和 `codebook_size`；`latents_to_codebook_vectors()` 不触发 dropout、loss 或 EMA 更新。
- `AutoGroupResidualVectorQuantizer` 从 `anytrain.module.quantization` 显式导入。每个 residual
  stage 使用两个同尺寸 learned group codebook；配置中的 `codebook_size` 是单个 group 的大小，
  对外 flat `codebook_size` 是它的平方，输出 `codebook_dim` 是配置值的两倍。`input_dim` 必须为
  偶数，推理可用 `num_active_codebooks` 截断 stage 数，训练 dropout 用 `indices == -1` 和
  `active_codebook_mask == False` 标记未启用 stage。
- 量化输出统一为 `QuantizeOutput`，离散整数用 `indices`，连续 codebook 空间向量用 `codebook_vectors`。
- RVQ dropout 下 inactive codebook 使用 `indices == -1` 和 `active_codebook_mask == False` 标记，不能只靠 `codebook_vectors` 判断有效性。

迁移设计见 [`docs/quantization-migration.md`](../quantization-migration.md)。第一版不迁入 audio codec、model zoo 或 deepaudio 的 projector/MoE 体系。

多模态 token id space 和 block embedding 路由已独立到 `anytrain.idspace`，
设计见 [`docs/modules/idspace.md`](idspace.md)。

## 依赖策略

`torch` 和 `einops` 是默认依赖。`einops.rearrange` 用于让动态分组卷积的 batch/channel/kernel shape 变换保持可读。

`anytrain.module` 后续如有额外三方依赖，使用 `module` extra 管理：

```bash
python -m pip install "anytrain[module]"
```

当前 Qwen3 复用层需要 `module` extra 中的 `transformers`。ADT、Dynamic Conv 和
quantization 不需要 optional 依赖；`import anytrain` 和 `import anytrain.module`
不会主动导入 `transformers`。
