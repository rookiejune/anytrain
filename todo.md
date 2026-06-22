# TODO

## 目标设计

`anytrain` 的核心不是一个大而泛的 task framework，也不是统一训练启动脚本，而是用户写 PyTorch/Lightning 训练代码，尤其是写普通 `LightningModule` 时会用到的组件库：

- `anytrain.lightning`：可复用的 LightningModule logging mixin 和训练调试 callback。
- `anytrain.loss` / `anytrain.evaluator` / `anytrain.optim` / `anytrain.module` / `anytrain.plotter` / `anytrain.framework`：用户写 LightningModule 时可选组合的组件。
- `anytrain.registry` / `anytrain.types`：轻量支撑层。

下游用户的主要工作流应该是：

1. 在自己的项目里定义配置系统和训练入口。
2. 写自己的 pl module、data module；模型、loss、evaluator、plotter 等子组件通过构造函数显式传入或在模块内创建。
3. 在自己的入口里创建 `Trainer` 并调用 `fit()`。

`anydataset` 负责数据集、canonical sample 和 batch dataclass；`anytrain` 不解释 batch schema，不内置具体任务 step。

Lightning 是核心依赖，测试默认要求安装 `torch` 和 `lightning`。Hydra、OmegaConf、pydantic、argparse 等配置工具由下游项目按需选择，不作为默认依赖。

## 期望使用方式

下游 pl module 直接继承 Lightning 原生基类：

```python
from lightning import pytorch as pl


class MyPLModule(pl.LightningModule):
    def __init__(self, model, loss):
        super().__init__()
        self.model = model
        self.loss = loss

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        output = self(batch.inputs)
        loss = self.loss(output, batch.targets)
        self.log("train/loss", loss)
        return loss
```

训练入口由下游项目自己维护：

```python
from lightning import pytorch as pl

from anytrain.lightning import StopOnNonfiniteLossCallback


def train():
    module = MyPLModule(model=MyModel(), loss=MyLoss())
    data_module = MyDataModule()
    trainer = pl.Trainer(
        default_root_dir="outputs/my_project/debug",
        callbacks=[StopOnNonfiniteLossCallback()],
    )
    trainer.fit(module, datamodule=data_module)
```

## 配置边界

`anytrain` 不定义顶层配置形状。推荐原则：

- 配置系统属于下游项目，可以用普通 Python、Hydra、pydantic、argparse 或其它项目内约定。
- `pl_module` 是训练语义入口；模型、loss、optimizer、scheduler、evaluator、plotter、framework 等组件通过下游 `pl_module.__init__` 显式接收，或由下游模块自己创建。
- optimizer / scheduler 由下游 `configure_optimizers()` 创建；直接返回 Lightning 原生结构。
- `Trainer`、logger、checkpoint resume 和运行目录由下游入口显式设置。

## Lightning 边界

`anytrain.lightning` 负责：

- LightningModule 侧核心 logging helper。
- non-finite loss 检查等训练调试 callback。

`anytrain.lightning` 不负责：

- 具体任务 step。
- 具体 data schema。
- 具体模型 registry。
- 配置装配或对象实例化。
- optional 领域组件的重依赖。

## 目标包结构

```text
src/anytrain/
  __init__.py
  lightning/
    __init__.py
    callback/
    mixin/
  loss/
  evaluator/
  optim/
  plotter/
  framework/
  module/
  registry.py
  types.py
examples/
  tiny_regression.py
```

说明：

- 仓库级 smoke/example 放在根目录 `examples/`。
- 下游项目自己的 `configs/` 不应该放进 `anytrain` 包里。
- `anytrain` 不提供 `python -m anytrain.*` 训练入口。

## 迁移清单

1. Done: 删除 `anytrain.hydra` 默认训练入口和 `hydra-core` 默认依赖。
2. Done: tiny regression smoke 示例改为下游自有 Python 入口，不放进 `src/anytrain` 包内。
3. Done: 将 LightningModule logging helper 明确为 core；第三方 backend 作为 optional。
4. Done: 将 loss/evaluator/plotter/framework 按 core/optional 子模块分层。
5. Done: 公开接口优先按 `pl_module` 组织，单一 `model` 只作为最简内部组件，不再作为硬性顶层字段。
6. Done: 从 `deepaudio.module.dynamic_conv` 迁入 1D Dynamic Conv / Dynamic Conv Transpose，并保留 einops 作为默认依赖以提高 shape 变换可读性。
7. Done: 按 `docs/quantization-migration.md` 从 `deepaudio.module.vector_quantizer` 迁入 task-agnostic 量化组件，第一版覆盖 FSQ、VQ、RVQ。
8. Done: 将 optimizer helper 从 `utils.optim` 提升到顶层 `anytrain.optim`。

## 已确认

- 不保留 `anytrain.hydra` 子包。
- 配置工具由下游选择；库内不依赖 Hydra/OmegaConf。
- `pl_module` 是主要训练语义入口，模型和其他组件由下游模块的 `__init__` 显式接收。
- optimizer / scheduler 不作为顶层绑定逻辑；下游 `pl_module` 自己决定配置形状和创建方式。
- `anydataset` 不作为强依赖，base package 里先不提供 integration。
- LightningModule logging helper 是 core；`wandb` 等第三方 backend 是 optional。
- loss/evaluator 有 core 接口和组合器；audio/text/speech/gan 等领域组件作为 optional 子模块。
- optimizer/scheduler helper 放在 `anytrain.optim`，但仍由下游 `pl_module.configure_optimizers()` 显式调用，不做隐藏注入。
- `module.dynamic_conv` 只迁入成熟的 1D 实现；2D dynamic conv 暂不迁入注释/实验代码。

## Quantization 迁移

设计文档：`docs/quantization-migration.md`

### P0: 设计冻结

- Done: 盘点 `deepaudio` quantization 候选源码和依赖边界。
- Done: 明确第一版不迁入 audio codec/model/zoo 和 projector/MoE 体系。
- Done: 固定目标包名为 `anytrain.module.quantization`。

### P1: FSQ 最小闭环

- Done: 新增 `src/anytrain/module/quantization/` 包结构。
- Done: 新增统一输出 dataclass：`QuantizationLoss`、`QuantizeOutput`。
- Done: 新增最小 projection helper，只支持 identity/linear。
- Done: 迁移 `FiniteScalarQuantizer` 和 `FSQConfig`。
- Done: 将 FSQ 配置字段从 `levels_per_codebook` 收敛为 `levels`。
- Done: 修正 FSQ `indices` 为单 codebook flat id，并保留 levels 转换 helper。
- Done: 修正 FSQ `codebook_size` 为 `prod(levels)`。
- Done: 去掉 FSQ 配置里的 `num_codebooks`，让输出 shape 和普通 VQ 对齐。
- Done: 对 even `levels` 增加 soft warning，推荐优先使用 odd levels 的对称网格。
- Done: 将 FSQ 默认 `levels` preset 改为 odd-only，避免默认构造触发 even-level warning。
- Done: 增加 FSQ `bound_scale`，用于缓解进入 `tanh` 前 latent 过大导致的边界饱和。
- Done: 增加 FSQ shape、round-trip、import smoke 测试。

### P2: VQ 迁移

- Done: 迁移 `VQConfig` 和 `EmbeddingVectorQuantizer`。
- Done: 修正 deepaudio 当前 `loss` / `vq_loss` 字段不一致。
- Done: 修正 EMA 和非 EMA 训练分支。
- Done: 增加 `normalize_latents: bool = True` 显式描述 l2-normalized nearest-neighbor lookup。
- Done: 增加 VQ loss、EMA、eval、backward 测试。

### P2.5: GVQ 迁移

- Done: 新增 `GVQConfig` 和 `GroupedVectorQuantizer`。
- Done: 用 `group_sizes` 表示 learned product codebook，例如 `(90, 90)` 对外是 `codebook_size=8100`。
- Done: 保持 GVQ 和 FSQ/VQ 一样的 flat `indices`、`codebook_vectors`、`quantized_latents` 接口。
- Done: 增加 group/flat indices round-trip、shape、gradient 和 lookup 测试。

### P3: RVQ 迁移

- Done: 迁移 `RVQConfig` 和 `ResidualVectorQuantizer`。
- Done: 将 RVQ `forward()` 收敛为返回 `QuantizeOutput`。
- Done: 第一版优先支持统一 `codebook_dim`，异构维度先明确报错。
- Done: 增加 RVQ residual、`num_active_codebooks`、dropout train/eval 测试。

### P4: 文档和公开导出

- Done: 更新 `docs/modules/module.md` 的当前状态。
- Done: 在 `anytrain.module.quantization.__init__` 导出稳定 API。
- Done: 在 `anytrain.module.__init__` 导出常用量化类。

## Evaluator Optional 迁移

设计文档：`src/anytrain/evaluator/todo.md`

### P0: Speech/Text 设计冻结

- Done: 补 `evaluator.text`，比较两个 text 的 BLEU/WER/chrF。
- Done: 补 `evaluator.speech` 的 Whisper ASR evaluator，ASR 后复用 text evaluator 计算 reference metrics。
- Done: 补 `evaluator.speech` 的 UTMOS evaluator，用显式注入 backend 封装 speech quality score。
- Done: 领域 evaluator 不进入 core import；缺 backend 时抛清晰错误。

### P1: Speech/Text 后端接入

- Todo: 按下游真实环境选择 Whisper backend 包装层，固定模型名、语言、device 和 decode options 配置入口。
- Todo: 按下游真实环境选择 UTMOS backend 包装层，固定 checkpoint/cache/device 配置入口。
- Todo: 如需切到 `sacrebleu` / `jiwer`，把依赖加入 `text` extra，并用当前测试锁住指标口径迁移。
