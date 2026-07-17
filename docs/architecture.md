# anytrain Architecture

## 定位

`anytrain` 是给用户编写 PyTorch/Lightning 训练代码时使用的组件库，重点服务普通 `LightningModule` 的实现。它提供下游 `LightningModule` 里可能用到的可组合积木，例如 logging mixin、debug callback、loss/evaluator 组合器、optimizer/scheduler helper 和 task-agnostic `torch.nn.Module`。

它的设计重点是帮助用户写自己的训练模块，而不是接管训练工程入口。下游项目自己决定配置系统、对象装配、运行目录、checkpoint resume 和 `Trainer.fit()` 调用；`anytrain` 不提供默认 CLI/app 层，也不要求下游继承自定义 `LightningModule` 基类。

核心目标：

- 提供写 `LightningModule` 时常用的 runtime helper。
- 提供可显式组合进训练模块的 loss/evaluator/logging/plotting/framework 组件。
- 提供 optimizer/scheduler 和 task-agnostic module 积木。
- 通过 optional dependencies 提供领域组件，而不是把所有重依赖压进 core。
- 保持入口和配置装配由用户项目显式拥有。

## 分层

### Core

默认安装必须可用：

- `anytrain.lightning`：LightningModule logging mixin 和 debug callback。
- `anytrain.loss`：通用 loss 接口和组合器。
- `anytrain.evaluator`：通用 evaluator 接口和组合器。
- `anytrain.optim`：optimizer、scheduler 和 LLM/Muon helper。
- `anytrain.module`：task-agnostic `torch.nn.Module` 积木。

Lightning、Torch 和 einops 是 core 依赖。`module` 顶层的 ADT、dynamic conv 和
quantization 默认可用；只有 Qwen3 builder 需要 `module` extra。Muon 是否可用取决于当前
PyTorch 是否提供 `torch.optim.Muon`，不影响其余 core API。

### Optional General

当前已实现的组件不绑定具体领域，但需要额外依赖：

- `plotter`：matplotlib/plotly/seaborn 等可视化。
- `chat`：环境变量驱动的大模型调用入口，用于训练工程里的实验总结、日志解释和 LLM 辅助评估原型。
- `tokenizer`：基于 Hugging Face `tokenizers` 的 codec frame BPE。
- third-party logger backend：wandb、mlflow 等。

规划中的 optional general 组件：

- metrics evaluator：基于 torchmetrics 的通用分类/回归指标。

### Optional Domain

这些组件由 `anytrain` 提供，但用户显式安装对应 extra：

- audio：spectral loss、codec wrapper、speech evaluator、audio plotter。
- text：文本生成/分类 evaluator。
- speech：WER/CER/ASR 相关 evaluator。
- TTS：公共协议保留轻量，具体 backend adapter 通过对应 extra 安装。

### Optional Framework

当前提供的 optional framework：

- flow matching。
- GAN adversarial objective、feature matching、WGAN-GP 和 DAC discriminator preset。

只有跨项目复用需求明确后再考虑：

- masked autoencoder。
- GAN manual optimization helper。
- 其他跨项目复用的训练范式。

这些不进入 core，避免 `anytrain` 的默认依赖和默认心智负担膨胀。

## 运行边界

`anytrain` 不定义固定运行链路。一个下游入口通常会自己完成：

1. 读取项目配置，或直接用 Python 构造对象。
2. 设置随机种子、matmul precision 等运行环境。
3. 创建下游 `LightningModule` 和 data module / dataloader。
4. 创建 `lightning.pytorch.Trainer`，设置 `default_root_dir`、logger、callback 和 resume 参数。
5. 调用 `trainer.fit(...)`。

这些步骤可以用普通 Python、Hydra、argparse、pydantic 或项目自己的配置系统实现。`anytrain` 只要求组件作为普通 Python 对象显式传入下游 `LightningModule`，不假设配置树形状，也不自动实例化对象。

## Lightning 接口

下游 pl module 直接继承 Lightning 原生基类：

```python
from lightning import pytorch as pl


class MyPLModule(pl.LightningModule):
    ...
```

`anytrain` 不接管组件注入。复杂项目可以把多个子模块、损失、优化器工厂、scheduler 工厂、辅助头、evaluator 或 plotter 直接声明为 `pl_module` 参数，也可以完全在代码中构造。

更完整的 Lightning 设计见 `docs/lightning.md`。逐模块设计文档见 `docs/modules/index.md`。

## 模块定位

### `lightning`

提供 LightningModule logging mixin 和 callback。logger backend 由下游或 Lightning 原生配置负责创建；第三方 logger backend 是 optional。

### `loss`

提供训练 step 中可直接使用的 loss 组件。core 里保留通用组合器；audio/text/speech 等领域 loss 通过 optional 子模块提供。GAN adversarial training 属于 `framework.gan`，不放在 `loss` 下。

### `evaluator`

提供 validation/test/training step 中可直接使用的 metric/evaluator 组件。core 里保留接口和组合器；codec/text/speech 等领域 evaluator 通过 optional 子模块提供。

### `optim`

提供 optimizer/scheduler helper。下游仍在自己的 `configure_optimizers()` 里显式调用 helper，并按 Lightning 原生格式返回。

### `module`

提供下游 LightningModule 可显式组合的 task-agnostic `torch.nn.Module` 积木，例如 Adaptive Dirichlet Tempering、1D Dynamic Conv 和量化模块。`einops` 是默认依赖，用于保持动态层 shape 变换可读；需要其它额外依赖的组件通过 `module` extra 暴露，不进入 package root import。

### `idspace` / `tokenizer`

`idspace` 是 core 的 local/global token id 映射和 embedding 路由。`tokenizer` 是 optional
general 算法层；CodecBPE 需要 `tokenizer` extra，不进入 package root import。

### `codec` / `tts`

`codec` 提供 optional audio codec wrapper。`tts` 的协议和输出类型保持轻量，具体 backend
adapter 按依赖 extra 隔离。两者都不接管数据集、训练 step 或模型组合。

### `plotter`

提供训练期可视化组件，通常依赖 `plot` extra。plotter 返回图形对象，logging 由下游 LightningModule 负责。

### `chat`

提供环境变量驱动的大模型调用入口，通常依赖 `chat` extra。它负责把显式 prompt 和实例内消息上下文发给指定 provider，不接管 prompt 模板、任务 schema、provider cache 观测或训练流程。

### `framework`

提供跨项目复用的训练范式，作为 optional/experimental 层，不进入 core。这里的 `framework` 是研究范式组件层，例如 flow matching、MAE 或 GAN helper；不是训练工程入口。

## 包结构

```text
src/anytrain/
  lightning/
    callback/
    mixin/
  loss/
  evaluator/
  optim/
  module/
    dynamic_conv/
    quantization/
  idspace/
  chat/
  plotter/
  tokenizer/
  codec/
  tts/
  framework/
  example/
```

`lightning`、`loss`、`evaluator`、`optim` 和 `module` 是核心体验；`chat`、`plotter`、`framework` 和领域组件按依赖拆分为 optional 子模块。

## 边界

`anytrain` 提供组件，但不做这些事：

- 替用户写完整 task module。
- 提供默认训练 CLI、app 层或配置装配协议。
- 替用户规定 batch schema。
- 把 optimizer/scheduler 作为顶层硬配置自动注入。
- 把模型 zoo 或预训练下载逻辑放进 core。
- 把所有 audio/text/speech/chat/plot/framework 依赖放进默认安装。
- 静默兼容缺失依赖；optional 子模块应明确提示安装对应 extra。
