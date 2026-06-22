# Components

## 设计原则

`anytrain` 的组件是给用户写普通 LightningModule 时使用的积木。Hydra app 层负责把这些组件按配置装配进下游对象；用户仍然控制训练逻辑，但可以选择 `anytrain` 提供的 module、loss、evaluator、plotter 和 framework。

组件应该是显式可组合的对象，而不是隐藏在自定义 `LightningModule` 基类里的行为。一个组件如果会改变训练流程、文件系统状态或日志行为，应通过明确的 helper、callback 或 `pl_module.__init__` 参数暴露。

组件按依赖拆分：

- core 组件默认可用。
- optional 组件在对应子模块中提供。
- optional 子模块缺依赖时应给出清晰错误，而不是影响 core import。

## Logging

LightningModule 侧 logging helper 是 core。

core 提供：

- experiment root 约定：`save_dir/name/version`。
- `Trainer.default_root_dir` 设置。
- `LightningLogMixin`：prefixed dict、audio 和 figure logging helper。

optional backend：

- wandb。
- mlflow。
- 其他第三方 logger。

第三方 backend 是 optional；logger backend 由下游或 Lightning 原生配置负责创建。

## Loss

core loss 负责统一训练 step 里的主 loss 约束：

```python
loss = loss_fn(prediction, target)
```

主返回必须是可用于 `backward()` 的 scalar Tensor。需要额外日志时，loss 可以返回 `(loss, details)`；`details` 是可选的 mapping，不参与反传。

建议接口：

- `LossABC`：抽象基类，子类实现 `compute_loss()`，统一校验 scalar 主 loss 和可选 details。
- `LossGroup`：用 mapping/`ModuleDict` 组合多个 loss，返回 `(total, details)`。
- `LossBalancerABC`：把多个命名 scalar loss 合成为一个 scalar total，可选返回 details；默认实现是 `MeanLossBalancer`，core 也提供 `UncertaintyLossBalancer`。
- `TaskLoss`：面向用户配置的组合容器，不绑定具体任务语义。

optional loss：

- spectral / temporal audio loss。
- GAN loss。
- text/speech 领域 loss。

## Evaluator

core evaluator 负责统一 metric 返回格式：

```python
metrics = evaluator(prediction, target)
```

建议接口：

- `EvaluatorABC`：继承 `torch.nn.Module` 的有状态 evaluator 抽象基类，子类实现 `evaluate()`。
- `EvaluatorGroup`：用 `nn.ModuleDict` 组合多个 evaluator，并处理 key 校验。
- `EvaluatorABC.update/compute/reset` 统一处理 epoch/running 状态生命周期，保持和 torchmetrics 风格兼容。

optional evaluator：

- audio/codec/speech evaluator。
- text evaluator。
- torchmetrics-backed 通用 evaluator。

## Module

`module` 是 optional general 组件，提供 task-agnostic 的 `torch.nn.Module` 积木。

适合放入：

- router/gating helper。
- 动态层。
- 不绑定 batch schema 或具体任务 step 的研究模块。

不适合放入：

- 完整模型 zoo。
- 项目私有网络结构。
- 需要 audio/text/speech 数据语义才能理解的模块。

当前提供：

- Adaptive Dirichlet Tempering，用于 MoE/router logits 的自适应温度缩放。
- 1D Dynamic Conv / Dynamic Conv Transpose，用于按样本或按分段组合 expert kernel；router 可注入，也可通过 `forward_manually()` 显式传入 expert 权重。

## Plotter

plotter 是 optional general 组件，依赖 `plot` extra。

建议边界：

- plotter 只负责从 tensor/state 生成 figure 或可记录对象。
- 下游 LightningModule 负责把图记录到 Lightning logger。
- audio/image/MoE 等具体 plotter 放 optional 子模块。

## Framework

framework 是 optional/experimental 组件。这里的 framework 指研究训练范式组件，不是默认 Hydra app 层。

适合放入：

- flow matching。
- masked autoencoder。
- GAN 训练辅助。

不适合放入 core，因为这些是训练范式，不是每个 LightningModule 都需要的基础能力。

不适合放入：

- 要求下游继承特定 `LightningModule` 基类的完整任务框架。
- 隐式接管 optimizer、scheduler、batch 解释或 logging 的大封装。
- 只能服务单个项目的私有训练模板。

## Import 规则

推荐规则：

- `import anytrain` 不导入 optional 领域依赖。
- `import anytrain.lightning` 可以假设 torch/lightning 已安装。
- `import anytrain.module` 可以假设 torch/einops 已安装；需要额外依赖的后续组件再要求 `module` extra。
- `import anytrain.loss.spectral` 和 `import anytrain.loss.temporal` 当前只依赖 core torch。
- `import anytrain.plotter` 可以要求 plot extra。
- optional 缺依赖时抛出明确错误，例如提示安装对应 extra。
