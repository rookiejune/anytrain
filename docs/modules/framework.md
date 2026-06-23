# `anytrain.framework` Design

## 定位

`anytrain.framework` 是 optional/experimental 研究框架层，用于沉淀跨项目复用的训练范式组件，例如 flow matching、masked autoencoder 或 GAN helper。

它不是 core 层，也不替下游项目生成完整 LightningModule。不要把这里的 `framework` 和训练工程入口混在一起：入口由下游项目自己维护，`anytrain.framework` 是可选训练范式积木。

## 当前状态

当前已有 `framework.flow_matching` 第一版 optional API，其他 framework 子模块仍只保留模块边界和 `todo.md`。任何新 framework 子模块都需要在复用需求明确后再迁入。

`framework.flow_matching` 的目标设计见 [`docs/flow-matching-design.md`](../flow-matching-design.md)。它应作为 Facebook `flow_matching` 的可组合再封装，提供 source、time sampler、objective 和 sampler 等积木，而不是完整任务框架。

## 适合放入的内容

- 可跨多个项目复用的训练目标或采样调度。
- 可组合进下游 LightningModule 的 helper。
- 明确依赖 extra 和测试边界的研究范式模块。
- 与 `loss`、`evaluator`、`plotter` 可组合但不强耦合的组件。

## 不适合放入的内容

- 完整任务工程模板。
- 要求下游继承特定 LightningModule 基类的大框架。
- 项目私有 batch schema。
- 项目私有模型 zoo。
- 数据集适配逻辑。
- 只在单个下游实验中使用的一次性代码。

## 依赖策略

framework 子模块默认不进入 core import。每个子模块需要声明自己的 extra 或安装说明，缺依赖时应直接报错并提示安装方式。

示例边界：

- `framework.flow_matching` 可以依赖 flow matching 所需的数值 helper，但不应依赖 audio/text 特定组件。
- `framework.gan` 可以提供 manual optimization helper，但具体 discriminator/generator 架构由下游项目提供。
- `framework.mae` 可以提供 mask 生成和重建损失辅助，但不接管完整数据 pipeline。

## 测试策略

每个 framework 子模块至少需要：

- 一个最小 CPU 单测，验证核心数学或状态转换。
- 一个与下游 LightningModule 可组合的 smoke test。
- optional 依赖缺失时的错误路径测试。
- 文档示例和测试配置保持一致。
