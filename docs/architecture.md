# anytrain Architecture

## 定位

`anytrain` 是 Hydra-first 的 Lightning 训练工程工具包。它提供一个清晰的 Hydra app 层来组织实验配置、对象装配、运行目录和 `Trainer.fit()`；同时提供用户编写普通 LightningModule 时常用的可组合训练组件。

它的设计重点是框架化训练工程，不框架化模型类。下游 `pl_module` 仍然直接继承 Lightning 原生 `LightningModule`，自己实现 `training_step()`、batch 解释和 optimizer/scheduler；`anytrain` 不提供魔法基类，也不隐藏注入组件。

核心目标：

- 统一 Hydra 启动入口和配置驱动装配流程。
- 统一 experiment root、resume 和运行审计边界。
- 提供 Lightning runtime helper。
- 提供通用 loss/evaluator/logging/plotting/framework 组件。
- 通过 optional dependencies 提供领域组件，而不是把所有重依赖压进 core。

## 分层

### Core

默认安装必须可用：

- `anytrain.hydra`：Hydra app 入口、配置驱动对象装配、Trainer 创建和 `fit()` 调用。
- `anytrain.lightning`：LightningModule logging mixin 和 debug callback。
- `anytrain.loss`：通用 loss 接口和组合器。
- `anytrain.evaluator`：通用 evaluator 接口和组合器。
- `anytrain.module`：task-agnostic `torch.nn.Module` 积木。
- `anytrain.registry` / `anytrain.types`：轻量支撑层。

Lightning 是 core 依赖，测试不应把 Lightning 当成 optional。

### Optional General

这些组件不绑定具体领域，但需要额外依赖：

- `module`：动态层、router/gating helper 等通用神经网络积木。
- `plotter`：matplotlib/plotly/seaborn 等可视化。
- metrics evaluator：基于 torchmetrics 的通用分类/回归指标。
- third-party logger backend：wandb、mlflow 等。

### Optional Domain

这些组件由 `anytrain` 提供，但用户显式安装对应 extra：

- audio：spectral loss、codec/speech evaluator、audio plotter。
- text：文本生成/分类 evaluator。
- speech：WER/CER/ASR 相关 evaluator。
- gan：GAN loss、D/G monitor、manual optimization helper。

### Optional Framework

研究框架可以作为 optional module 提供：

- flow matching。
- masked autoencoder。
- 其他跨项目复用的训练范式。

这些不进入 core，避免 `anytrain` 的默认依赖和默认心智负担膨胀。

## 运行链路

`python -m anytrain.hydra --config-dir configs --config-name train` 调用 `anytrain.hydra.run_train(cfg)`。

`run_train()` 是 `anytrain` 的核心 Hydra app 闭环。它的顶层配置分成模块槽位和运行 envelope。模块槽位只有 `pl_module`、`data_module` 和 `trainer`；`environment`、`experiment`、`fit`、`print_config` 只描述运行环境和启动参数。

固定流程是：

1. `validate_train_config(cfg)`：校验模块槽位和运行 envelope。
2. `configure_environment(cfg.environment)`：设置 matmul precision 和随机种子。
3. `instantiate_train_modules(cfg)`：实例化 `pl_module`、`data_module` 和 `trainer`。
4. `trainer.fit(lightning_module, datamodule=data_module, ckpt_path=cfg.fit.ckpt_path)`。

`anytrain.hydra` 不理解 batch、loss、metric 或 task 语义。它只负责把用户声明的对象装起来并交给 Lightning。

## Hydra 框架边界

`anytrain` 的框架感应该集中在 Hydra app 层：

- 定义稳定的顶层配置形状：模块槽位 `pl_module`、`data_module`、`trainer`，运行 envelope `environment`、`experiment`、`fit`、`print_config`。
- 用 Hydra instantiate 装配用户声明的对象，并让对象依赖通过 `pl_module.__init__` 显式暴露。
- 统一 `experiment.save_dir/name/version` 和 `Trainer.default_root_dir`。
- 明确 checkpoint resume 入口，例如 `fit.ckpt_path`。
- 对错误配置尽早失败，例如缺少 `pl_module`、顶层出现 `optimizer` / `scheduler` 或 logger backend 配置不清晰。

`anytrain` 的框架层不做这些事：

- 不要求下游继承 `AnyTrainModule`。
- 不替用户实现 task-level step。
- 不从 batch schema 猜测 loss、metric 或模型输入。
- 不把 optimizer/scheduler 作为顶层硬配置注入到模型里。
- 不把项目私有数据适配、模型 zoo 或预训练下载协议放进 core。

## 配置接口

最小配置形状：

```yaml
pl_module:
  _target_: my_project.pl_modules.MyPLModule
  model:
    _target_: my_project.models.MyModel
  loss_fn:
    _target_: anytrain.loss.LossGroup
    losses:
      reconstruction:
        _target_: torch.nn.L1Loss
  optimizer:
    _target_: torch.optim.AdamW
    _partial_: true
    lr: 0.0003

data_module:
  _target_: my_project.data.MyDataModule

experiment:
  save_dir: outputs
  name: my_project
  version: debug

trainer:
  max_epochs: 1
  accelerator: auto
  devices: auto
  logger: true

fit:
  ckpt_path: null
```

`pl_module` 指向下游 Lightning module。模型、loss、optimizer、scheduler、evaluator、plotter 等组件都作为 `pl_module.__init__` 的显式参数进入，或由下游模块自己创建；`anytrain` 不提供隐藏注入层。

## Lightning 接口

下游 pl module 直接继承 Lightning 原生基类：

```python
from lightning import pytorch as pl


class MyPLModule(pl.LightningModule):
    ...
```

`anytrain` 不接管组件注入。复杂项目可以把多个子模块、损失、优化器工厂、scheduler 工厂、辅助头、evaluator 或 plotter 直接声明为 `pl_module` 参数。

更完整的 Lightning 设计见 `docs/lightning.md`。逐模块设计文档见 `docs/modules/index.md`。

## 模块定位

### `hydra`

承接 Hydra app：训练入口、环境配置、Hydra instantiate、Trainer 创建、运行 root 设置和 `fit()` 调用。

### `lightning`

提供 LightningModule logging mixin 和 callback。logger backend 由下游或 Lightning 原生配置负责创建；第三方 logger backend 是 optional。

### `loss`

提供训练 step 中可直接使用的 loss 组件。core 里保留通用组合器；audio/gan 等领域 loss 通过 optional 子模块提供。

### `evaluator`

提供 validation/test/training step 中可直接使用的 metric/evaluator 组件。core 里保留接口和组合器；codec/text/speech 等领域 evaluator 通过 optional 子模块提供。

### `module`

提供下游 LightningModule 可显式组合的 task-agnostic `torch.nn.Module` 积木，例如 Adaptive Dirichlet Tempering 和 1D Dynamic Conv。`einops` 是默认依赖，用于保持动态层 shape 变换可读；需要其它额外依赖的组件通过 `module` extra 暴露，不进入 package root import。

### `plotter`

提供训练期可视化组件，通常依赖 `plot` extra。plotter 返回图形对象，logging 由下游 LightningModule 负责。

### `framework`

提供跨项目复用的训练范式，作为 optional/experimental 层，不进入 core。

这里的 `framework` 是研究范式组件层，例如 flow matching、MAE 或 GAN helper；不要和 Hydra app 层混在一起。Hydra app 是默认训练工程入口，`anytrain.framework` 是可选训练范式积木。

### `utils`

只放跨模块稳定复用的小工具。能放到具体模块里的 helper 不放进 `utils`。

## 包结构

```text
src/anytrain/
  hydra/
    app.py
    environment.py
    instantiate.py
    paths.py
    trainer.py
  lightning/
    callback/
    mixin/
  loss/
  evaluator/
  plotter/
  framework/
  module/
    dynamic_conv/
  registry.py
  types.py
```

`hydra` 和 `lightning` 是核心体验；`loss`、`evaluator`、`plotter`、`framework` 是训练组件层，按依赖拆分 core 与 optional 子模块。

## 边界

`anytrain` 提供组件，但不做这些事：

- 替用户写完整 task module。
- 替用户规定 batch schema。
- 把模型 zoo 或预训练下载逻辑放进 core。
- 把所有 audio/text/speech/plot/framework 依赖放进默认安装。
- 静默兼容缺失依赖；optional 子模块应明确提示安装对应 extra。
