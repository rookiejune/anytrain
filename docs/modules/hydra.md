# `anytrain.hydra` Design

## 定位

`anytrain.hydra` 是 Hydra-first 的训练工程入口。它负责把下游声明的训练模块装配起来，创建 Lightning `Trainer`，设置 experiment root，并调用 `trainer.fit()`。

这是 `anytrain` 的框架层：框架化配置组合、对象装配、运行目录和启动流程；不框架化 `LightningModule` 本身。这个模块不理解任务语义，不解释 batch，也不替下游创建 optimizer / scheduler。

## 当前实现

公开函数：

- `configure_environment(cfg)`：设置 `torch_matmul_precision` 和随机种子。
- `create_trainer(cfg, *, experiment=None)`：将 trainer 配置转换为 `lightning.pytorch.Trainer`。
- `validate_train_config(cfg)`：校验顶层配置边界。
- `instantiate_train_modules(cfg)`：实例化 `pl_module`、`data_module` 和 `trainer`。
- `run_train(cfg)`：执行完整训练装配流程并返回 `(trainer, lightning_module)`。
- `main(cfg)`：Hydra CLI 入口，支持 `python -m anytrain.hydra`。

源码拆分：

- `app.py`：`run_train()` 和 Hydra CLI `main()`。
- `config.py`：配置边界校验和 `Trainer.fit()` 参数提取。
- `environment.py`：环境配置，例如 seed 和 matmul precision。
- `modules.py`：训练模块槽位实例化。
- `trainer.py`：Trainer 创建、root 设置和 callback 装配。
- `instantiate.py`：Hydra instantiate 相关 helper。
- `paths.py`：experiment root 和 logger 字段。

## 配置输入

Hydra app 层的顶层配置分两类。

模块槽位是对象图的最小闭环：

- `pl_module`：必须提供，必须是带 `_target_` 的 Hydra object config。
- `data_module`：可选；提供时必须是带 `_target_` 的 Hydra object config。
- `trainer`：可选；是 Lightning `Trainer` 的 keyword arguments，不写 `_target_`。

运行 envelope 只描述运行环境和启动参数，不参与对象依赖图：

- `environment`
- `experiment`
- `fit`
- `print_config`

推荐的最小配置形状：

```yaml
environment:
  seed: 0
  seed_workers: true
  torch_matmul_precision: medium

experiment:
  save_dir: outputs
  name: anytrain
  version: debug

pl_module:
  _target_: my_project.pl_modules.MyPLModule
  model:
    _target_: my_project.models.MyModel
  optimizer:
    _target_: torch.optim.AdamW
    _partial_: true
    lr: 0.0003

data_module:
  _target_: my_project.data.MyDataModule

trainer:
  max_epochs: 10
  accelerator: auto
  devices: auto
  logger: true

fit:
  ckpt_path: null
```

`pl_module` 是必须字段。模型、loss、optimizer、scheduler、evaluator、plotter、framework 等组件都通过下游 `pl_module.__init__` 的显式参数传入，或者由下游 LightningModule 自己创建。

不新增顶层 `model`、`loss`、`optimizer` 或 `scheduler` 这类硬字段。复杂项目也应该把多个模型、辅助头、loss、optimizer、scheduler、evaluator 或 plotter 明确声明在 `pl_module` 配置下，让构造函数签名成为真实接口。

## 运行流程

`run_train(cfg)` 的固定顺序：

1. 调用 `validate_train_config(cfg)` 校验模块槽位和运行 envelope。
2. 可选打印 Hydra 配置。
3. 调用 `configure_environment(cfg.environment)`。
4. 调用 `instantiate_train_modules(cfg)` 实例化 `pl_module`、`data_module` 和 `trainer`。
5. 调用 `trainer.fit(lightning_module, datamodule=data_module, ckpt_path=cfg.fit.ckpt_path)`。

## Logger 与 Root

`experiment.save_dir/name/version` 是统一 root 约定：

```text
${save_dir}/${name}/${version}
```

`create_trainer()` 会用这组字段设置 `Trainer.default_root_dir`。logger backend 不由 `anytrain` 自动创建；需要 logger 时，直接使用 Lightning 原生 `trainer.logger` 参数或在 Python 代码中传入 logger 实例。当 `trainer.logger: false` 时关闭 logger。

`trainer.logger` 不接受 Hydra config object。第三方 logger backend 应由 Python 代码传入实例，或后续通过明确的 optional backend API 支持。

## 错误策略

- 缺少 `pl_module` 时抛出 `ValueError`。
- `pl_module` / `data_module` 不是 Hydra object config 时抛出 `ValueError`。
- `trainer` 写成 `_target_` object config 时抛出 `ValueError`，因为这个槽位只接收 `Trainer` kwargs。
- 顶层出现 `model`、`loss`、`optimizer`、`scheduler`、`evaluator` 等 `pl_module` 依赖时抛出 `ValueError`。
- `fit` 出现当前不支持的字段时抛出 `ValueError`。
- `trainer.logger` 是 dict/config 时抛出 `ValueError`，避免把 logger backend 装配规则藏进通用入口。

## 边界

`anytrain.hydra` 不做：

- 不实现 `training_step` / `validation_step` / `test_step`。
- 不解析 batch schema。
- 不注册模型 zoo。
- 不迁移 `anydataset` 的数据适配规则。
- 不把 optimizer / scheduler 做成顶层硬配置。
- 不提供或要求继承 `AnyTrainModule` 这类自定义基类。

## 测试策略

当前覆盖：

- `tests/test_main.py`：Hydra 风格配置可完整跑通 tiny regression。
- `tests/test_lightning.py`：`create_trainer()` 能实例化 callback 配置。

后续如果扩展 `anytrain.hydra`，需要优先补齐配置错误、root 设置、checkpoint resume 和 callback 装配的单测。
