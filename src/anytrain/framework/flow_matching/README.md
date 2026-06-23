# Flow Matching

`anytrain.framework.flow_matching` 目前只保留实现目录。目标设计见 [`docs/flow-matching-design.md`](../../../../docs/flow-matching-design.md)。

第一版会作为 Facebook `flow_matching` 的可组合再封装，优先提供 source、time sampler、objective、sampler 和 continuous/discrete preset，不接管下游 `LightningModule` 或 batch schema。

连续 preset 默认使用 `LogitNormalTimeSampler(t_min=0.0, t_max=1.0)`；离散 preset 默认使用 `LogitNormalTimeSampler(t_min=0.0, t_max=1.0 - 1e-3)`，和离散 Euler sampler 的端点保持一致。离散输入必须是 `torch.long` token tensor。

`Objective` 是底层训练目标入口，`FlowMatcher` 是 preset 组合器入口；matcher 不接收 objective 作为构造参数。
