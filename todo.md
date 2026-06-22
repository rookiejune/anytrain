# TODO

## 目标设计

`anytrain` 的核心不是一个大而泛的 task framework，也不是只做启动脚本，而是：

- `anytrain.hydra`：Hydra 配置驱动的训练装配入口。
- `anytrain.lightning`：可复用的 LightningModule logging mixin 和训练调试 callback。
- `anytrain.loss` / `anytrain.evaluator` / `anytrain.plotter` / `anytrain.framework`：用户写 LightningModule 时可选使用的训练组件层。

下游用户的主要工作流应该是：

1. 在自己的项目里定义 `configs/`。
2. 写自己的 pl module、data module；模型、loss、evaluator、plotter 等子组件尽量挂在 `pl_module` 配置里。
3. 通过 `anytrain.hydra` 的训练入口读取 Hydra 配置并启动训练。

`anydataset` 负责数据集、canonical sample 和 batch dataclass；`anytrain` 不解释 batch schema，不内置具体任务 step。

Lightning 是核心依赖，测试默认要求安装 `torch`、`lightning`、Hydra 相关依赖。

## 期望使用方式

命令行入口保持 Hydra 风格：

```bash
python -m anytrain.hydra --config-dir configs --config-name train
```

Python 入口：

```python
from omegaconf import OmegaConf
from anytrain.hydra import run_train

cfg = OmegaConf.load("configs/train.yaml")
run_train(cfg)
```

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

`anytrain.hydra` 装配时直接实例化 `cfg.pl_module`，Hydra 会递归装配它内部的模型或其他组件。

## 配置约定

推荐下游配置树：

```text
configs/
  train.yaml
  environment/
    default.yaml
  pl_module/
    default.yaml
  trainer/
    default.yaml
```

最小 `train.yaml` 形状：

```yaml
defaults:
  - environment: default
  - pl_module: default
  - trainer: default
  - _self_

experiment:
  save_dir: outputs
  name: debug
  version: v0

fit:
  ckpt_path: null

print_config: true
```

关键 group 约定：

- `pl_module`：必须指向下游 Lightning module；它可以在自己的 config 里显式接收一个或多个模型/loss/evaluator/plotter/framework 组件。
- `data_module`：由下游提供，可以是 LightningDataModule，也可以是可被 Trainer 接收的 datamodule。
- optimizer / scheduler：由下游 `pl_module` 自己配置和创建；直接返回 Lightning 原生结构。
- `trainer`：直接透传给 `lightning.pytorch.Trainer`；logger backend 由下游配置或 Lightning 原生机制决定；关闭日志使用 `trainer.logger: false`。

## Hydra 边界

`anytrain.hydra` 负责：

- 打印/解析配置。
- 设置环境，例如 seed、matmul precision。
- 实例化顶层 `pl_module`，并递归装配它内部需要的组件。
- 通过 Hydra 直连实例化可选的 `data_module`。
- 实例化 Trainer。
- 调用 `trainer.fit(...)`。

`anytrain.hydra` 不负责：

- 写具体任务语义或 `training_step`。
- 解释 batch 内容。
- 注册或加载项目特定模型 zoo。
- 写数据集适配规则。
- 把 audio codec、source separation、text-to-audio 等任务 step 写进 core。

## Lightning 边界

`anytrain.lightning` 负责：

- LightningModule 侧核心 logging helper。
- non-finite loss 检查等训练调试 callback。

`anytrain.lightning` 不负责：

- 具体任务 step。
- 具体 data schema。
- 具体模型 registry。
- optional 领域组件的重依赖。

## 目标包结构

```text
src/anytrain/
  __init__.py
  hydra/
    __init__.py
    __main__.py
    app.py
    environment.py
    instantiate.py
    paths.py
    trainer.py
  lightning/
    __init__.py
    callback/
    mixin/
  loss/
  evaluator/
  plotter/
  framework/
  registry.py
  types.py
examples/
  tiny_regression.py
  configs/
```

说明：

- `hydra/` 放 `run_train`、运行时 helper 和 Hydra 命令行入口。
- 仓库级 smoke/example 放在根目录 `examples/`。
- 下游项目自己的 `configs/` 不应该放进 `anytrain` 包里。

## 迁移清单

1. 文档先行：用本文件和 `docs/architecture.md` 固定 `hydra + lightning` 设计。
2. 将训练入口整理到 `src/anytrain/hydra/`。
3. 把 `run_train` 和有额外运行时语义的 helper 拆到 `hydra/` 子模块。
4. 把 Hydra CLI 入口改为：

   ```bash
   python -m anytrain.hydra
   ```

5. 更新测试，统一从 `anytrain.hydra` 进入。
6. 更新 README / AGENTS / architecture，统一 `hydra + lightning` 作为正式接口。
7. tiny regression smoke 示例放在仓库根目录的 `examples/`，不放进 `src/anytrain` 包内。
8. 将配置字段从 `task` 统一为 `pl_module`。
9. 删除 base package 里的 `integrations/anydataset`，数据连接留给下游项目。
10. 将 LightningModule logging helper 明确为 core；第三方 backend 作为 optional。
11. 将 loss/evaluator/plotter/framework 按 core/optional 子模块分层。
12. 公开接口优先按 `pl_module` 组织，单一 `model` 只作为最简内部组件，不再作为硬性顶层字段。
13. 从 `deepaudio.module.dynamic_conv` 迁入 1D Dynamic Conv / Dynamic Conv Transpose，并保留 einops 作为默认依赖以提高 shape 变换可读性。
14. 按 `docs/quantization-migration.md` 从 `deepaudio.module.vector_quantizer` 迁入 task-agnostic 量化组件，第一版覆盖 FSQ、VQ、RVQ。

## 已确认

- `hydra` 使用子包拆分，不保留顶层 `api.py`。
- 配置里统一使用 `pl_module`，不把用户自定义 task 语义占掉。
- `pl_module` 是主要装配入口，模型和其他组件由下游模块的 `__init__` 显式接收。
- optimizer / scheduler 不作为 `anytrain.hydra` 顶层绑定逻辑；下游 `pl_module` 自己决定配置形状和创建方式。
- `anydataset` 不作为强依赖，base package 里先不提供 integration。
- LightningModule logging helper 是 core；`wandb` 等第三方 backend 是 optional。
- loss/evaluator 有 core 接口和组合器；audio/text/speech/gan 等领域组件作为 optional 子模块。
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
