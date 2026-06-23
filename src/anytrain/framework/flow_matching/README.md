# Flow Matching

`anytrain.framework.flow_matching` 目前只保留实现目录。目标设计见 [`docs/flow-matching-design.md`](../../../../docs/flow-matching-design.md)。

第一版会作为 Facebook `flow_matching` 的可组合再封装，优先提供 source、time sampler、objective、sampler 和 continuous/discrete preset，不接管下游 `LightningModule` 或 batch schema。

默认 time sampler 是 `LogitNormalTimeSampler(t_min=1e-3, t_max=1.0 - 1e-3)`；需要完整区间时可以显式传 `UniformTimeSampler`。
