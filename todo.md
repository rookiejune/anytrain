# TODO

只记录尚未实现或需要真实环境确认的需求。历史迁移记录以文档和 git 历史为准。

## Loss

- 实现 `TaskLoss`：plain-config 友好的组合容器，不绑定具体任务语义。
- 增加更多 balancer 策略，例如 deviation。
- `framework.gan` 后续补 manual optimization helper；通用 adversarial objective 已不放在 `loss` 下。
- 只有在依赖和复用需求明确后，再加入 `loss.text` / `loss.speech`。

## Evaluator

- 根据下游真实使用情况决定是否增加常用 stateful evaluator，例如 MAE、MSE、accuracy。
- 增加 `evaluator.metrics`：torchmetrics-backed 通用分类/回归 evaluator，放 optional 子模块而不是 core。
- 增加 `evaluator.audio`：codec、speech 或 audio quality evaluator。
- 按下游真实环境选择 Whisper backend 包装层，固定模型名、语言、device 和 decode options 配置入口。
- 按下游真实环境选择 UTMOS backend 包装层，固定 checkpoint、cache、device 和 batch 推理行为。
- 如需切到 `sacrebleu` / `jiwer`，把依赖加入 `text` extra，并用当前测试锁住 public key 和空 reference 策略。

## Module

- Qwen3：根据真实训练需求暴露 SDPA / flash-attention knobs。
- Qwen3：只有在 `anycodec` 需要比 Hugging Face `Qwen3Model` 更窄的契约时，再增加 codec-specific token-forward wrapper。
- Qwen3：KV cache 语义需要独立设计，不混进训练 forward。
- Adaptive Dirichlet Tempering：支持非均匀 expert prior，用于 expert capacity 或有意偏置的 routing target。
- Adaptive Dirichlet Tempering：增加 top-k / hard routing 支持，包括 gumbel top-k 和 straight-through 变体。
- Adaptive Dirichlet Tempering：暴露可选辅助 loss，例如 KL-to-prior、entropy bonus 或 load-balance loss，但不自动接进 task loss。
- Adaptive Dirichlet Tempering：扩展 `diagnostics()`，加入 entropy、perplexity、min/max load、dead expert count 和 load imbalance ratio，并补 skewed routing 测试。

## Framework

- 后续新增 framework 子模块时，需要明确依赖 extra 和最小测试。
- masked autoencoder 等训练范式只有在跨项目复用明确后再迁入。
- `framework.gan` 后续根据真实复用需求补 manual optimization helper 和 audio discriminator。
- framework 不替用户写完整 `pl_module`，只提供训练逻辑可复用组件。

## Descript Audio Codec

- 在预置官方 checkpoint 的环境里用真实短音频跑一次 encode/decode smoke，确认
  `DAC.from_pretrained(local_files_only=True)`、统一 codes 轴和多 `n_quantizers` 配置。
