# TODO

framework 是 optional/experimental 层，不进入 core。

1. 先保留目录边界，不把 flow matching / MAE / GAN helper 放进默认 import。
2. 每个 framework 子模块需要明确依赖 extra 和最小测试。
3. 只有跨项目复用明确后再从下游项目迁入。
4. framework 不替用户写完整 `pl_module`，只提供训练逻辑可复用组件。

## Flow Matching

设计文档：[`docs/flow-matching-design.md`](../../../docs/flow-matching-design.md)

### P0: 设计冻结

- Done: 参考 deepaudio 的 continuous/discrete helper 和 sound_flow 使用方式。
- Done: 明确第一版是 Facebook `flow_matching` 的可组合再封装，不迁入 audio/text/codec 任务逻辑。
- Done: 固定 source、time sampler、objective、sampler、preset 的组件边界。

### P1: 依赖和基础组件

- Done: 新增 `anytrain.framework.flow_matching` 包结构和 `_deps.py` optional 依赖边界。
- Done: 增加 `flow` optional extra，安装 Facebook `flow_matching`。
- Done: 实现 source/time/model caller 基础组件。
- Done: 增加 core import 不触发 optional 依赖的 smoke test。

### P2: Continuous 最小闭环

- Done: 实现 continuous objective、ODE sampler 和 `ContinuousFlowMatcher` preset。
- Done: 增加 CPU toy model 的 loss、backward 和 sample shape 测试。

### P3: Discrete 最小闭环

- Done: 实现 discrete generalized KL objective、Euler sampler 和 `DiscreteFlowMatcher` preset。
- Done: 增加 uniform/mask source、loss 和 sample shape 测试。
