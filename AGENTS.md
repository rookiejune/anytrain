# anytrain

这是从 `deepaudio.lightning` 和训练组件中抽出的 PyTorch/Lightning 训练代码组件库，不再绑定 audio。它重点服务用户自己写普通 `LightningModule` 时会用到的组件，不框架化用户自己的入口、配置系统或 LightningModule。

## 项目边界

- 数据、canonical sample 和 batch schema 由外部 `anydataset` 或下游项目负责；`anytrain` 不内置数据集适配规则。
- 具体 task 语义、`training_step`、batch 解释和模型组合由下游用户实现。
- 训练入口、配置装配、运行目录和 `fit()` 调用由下游项目自己定义；`anytrain` 不提供默认 app 层。
- `lightning` 负责 task-agnostic 的 LightningModule logging mixin 和训练调试 callback；Lightning 是核心依赖，不是 optional integration。
- 公开接口优先围绕 `pl_module`，不要把单一 `model` 写成硬边界。
- 下游 `pl_module` 直接继承 Lightning 原生 `LightningModule`；不要提供或要求继承 `AnyTrainModule` 这类魔法基类。
- `loss`、`evaluator`、`optim`、`module`、`plotter`、`framework` 是下游训练模块可显式组合的组件，按 core/optional 子模块拆分依赖。
- `registry.py`、`types.py` 是轻量支撑层。

## 第一版保留

- `lightning`
- `loss`
- `evaluator`
- `plotter`
- `framework`
- 从 `deepaudio.protocol` 中抽出的通用基础类型和 registry 边界

## 第一版避免

- 不迁移 `datasets/`、`data_module/`、具体 `task/`、`wrapper/`、`zoo/`、`_pretrained/`。
- 不把 optimizer / scheduler / batch schema 做成隐藏注入协议。
- 不把 audio codec、source separation、text-to-audio 等任务 step 放进 core。
- 不把 `anydataset` 的数据集适配规则复制进来；数据依赖留给下游项目。
- 不把需要领域依赖的 loss/evaluator/plotter 放进 core；放 optional 子模块。

Lightning 边界见 `docs/lightning.md`，组件分层见 `docs/components.md`，正式架构见 `docs/architecture.md`，逐模块设计见 `docs/modules/index.md`，待办拆分见 `todo.md`。
