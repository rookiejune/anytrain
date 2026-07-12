# Flow Matching

`anytrain.framework.flow_matching` 目前只保留实现目录。目标设计见 [`docs/flow-matching-design.md`](../../../../docs/flow-matching-design.md)。

第一版会作为 Facebook `flow_matching` 的可组合再封装，优先提供 source、time sampler、runtime、objective 和 sampler，不接管下游 `LightningModule` 或 batch schema。

连续 runtime 默认使用 `UniformTimeSampler(t_min=0.0, t_max=1.0)`；离散 runtime 默认使用 `UniformTimeSampler(t_min=0.0, t_max=1.0 - 1e-3)`，和离散 Euler sampler 的端点保持一致。`LogitNormalTimeSampler` 保留为显式实验选项。离散输入必须是 `torch.long` token tensor。

`ContinuousFlowRuntime` 和 `DiscreteFlowRuntime` 使用同一套 API，持有 path、source、训练时间采样器、generation sampler 和 model caller。objective 只接收对应 runtime；下游需要自己的 mask 和 reduction 时可直接使用 `runtime.training_sample()`。

统一入口为 `objective(model, target, ...)` 和 `runtime.sample(model, source, ...)`。
