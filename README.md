# anytrain

`anytrain` 是 Hydra-first 的 PyTorch/Lightning 训练工程工具包。它负责把配置驱动的对象装配、experiment root/logger、checkpoint resume 入口和 `Trainer.fit()` 启动放在一个清晰的 Hydra app 层里；模型训练逻辑仍由下游普通 `LightningModule` 显式实现。

它的核心边界是：框架化实验组织，不框架化模型本身。`anytrain` 不提供也不要求继承自定义 `LightningModule` 基类；它提供编写 `pl_module` 时有用的 loss/evaluator/logger/debug callback 等训练组件，供下游按需组合。

边界约定：

- Hydra app 层是核心体验：配置组合、对象实例化、运行目录和 `fit()` 调用由 `anytrain.hydra` 统一承接。
- 数据、canonical sample 和 batch schema 由 `anydataset` 或下游项目负责，`anytrain` 不内置数据适配规则。
- 下游项目负责具体任务语义、`training_step`、batch 解释和模型组合。
- `anytrain.hydra` 负责按 Hydra 配置装配 `pl_module`、data module 和 trainer，并按 experiment 字段设置 `Trainer.default_root_dir`；logger 由下游配置或 Lightning 原生机制决定。
- `anytrain.lightning` 负责提供 task-agnostic 的 LightningModule logging mixin 和训练调试 callback；Lightning 是核心依赖，不是 optional integration。
- `anytrain.loss`、`anytrain.evaluator`、`anytrain.plotter`、`anytrain.framework` 是可组合训练组件层，按 core/optional 子模块拆分依赖。

## Quick Start

Hydra 入口是 `anytrain` 的默认训练入口，但配置由下游项目自己提供：

```bash
python -m anytrain.hydra --config-dir configs --config-name train
```

下游项目通常只需要写自己的 Lightning module，然后在 YAML 中指定 `_target_`：

```yaml
pl_module:
  _target_: my_project.pl_modules.MyPLModule
  model:
    _target_: my_project.models.MyModel
    hidden_dim: 256
  lr: 0.0003

data_module:
  _target_: my_project.data.MyDataModule
  batch_size: 32

experiment:
  save_dir: outputs
  name: my_project
  version: debug

trainer:
  max_epochs: 10
  accelerator: auto
  devices: auto
  callbacks:
    - _target_: anytrain.lightning.StopOnNonfiniteLossCallback
```

这是最简的单模型写法；更复杂的项目可以把多个组件直接挂在 `pl_module` 配置下，而不是把 `model` 提升成单独的顶层硬字段。

`pl_module` 代码直接继承 Lightning 原生基类：

```python
import torch
import torch.nn.functional as F
from lightning import pytorch as pl


class RegressionPLModule(pl.LightningModule):
    def __init__(self, model: torch.nn.Module, *, lr=0.0003):
        super().__init__()
        self.model = model
        self.lr = lr

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = F.mse_loss(pred, y)
        self.log("train/loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
```

Hydra 装配时，`anytrain.hydra` 会直接实例化 `cfg.pl_module`，并递归装配它内部需要的组件。optimizer / scheduler 由下游 `pl_module.configure_optimizers()` 自己创建；多 optimizer、多模块参数组等复杂逻辑直接按 Lightning 原生写法返回。

仓库内 smoke 示例放在 `examples/`，不进入 `src/anytrain` 包：

```bash
PYTHONPATH=src python -m anytrain.hydra --config-dir examples/configs --config-name tiny_regression
```

## 目标 v0

计划保留：

- `anytrain.hydra`：Hydra-first 训练入口、配置驱动对象装配、运行 root 设置和 `fit()` 调用。
- `anytrain.lightning`：LightningModule logging mixin 和 task-agnostic callback。
- `anytrain.loss`：通用 loss 接口和组合器；领域 loss 放 optional 子模块。
- `anytrain.evaluator`：通用 evaluator 接口和组合器；audio/text/speech evaluator 放 optional 子模块。
- `anytrain.module`：task-agnostic `torch.nn.Module` 积木，例如 ADT 和 1D dynamic conv。
- `anytrain.plotter`：训练期可视化组件，依赖 `plot` extra。
- `anytrain.framework`：可选研究框架层，例如 flow matching / MAE / GAN helper。
- `anytrain.registry` / `anytrain.types`：轻量 registry 和自动命名枚举。

明确不做：

- 不提供 `AnyTrainModule` 这类魔法继承基类。
- 不把 optimizer / scheduler / batch schema 做成隐藏注入协议。
- 不把项目私有 task、model zoo 或数据适配规则放进 core。

Lightning 边界见 `docs/lightning.md`，组件分层见 `docs/components.md`，总体架构见 `docs/architecture.md`，逐模块设计见 `docs/modules/index.md`，迁移清单见 `todo.md`。
