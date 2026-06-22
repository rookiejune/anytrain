# `anytrain.optim` Design

## 定位

`anytrain.optim` 提供 optimizer / scheduler 构造 helper。它不接管训练流程，也不要求下游继承特殊 LightningModule 基类；下游仍然在自己的 `configure_optimizers()` 里显式调用这些 helper。

## 当前结构

源码结构：

```text
src/anytrain/optim/
  __init__.py
  adamw.py
  compose.py
  config.py
  llm.py
  muon.py
  rules.py
  scheduler.py
```

当前公开导出：

- `AdamWConfig`
- `AdamWDecayPolicy`
- `MuonConfig`
- `MuonAdjustLRFn`
- `MuonAdamWConfig`
- `CompositeOptimizer`
- `ExcludedModules`
- `ExcludedModuleTypes`
- `create_adamw_optimizer`
- `split_adamw_decay_params`
- `create_muon_adamw_optimizer`
- `split_muon_params`
- `CurveShape`
- `SchedulerConfig`
- `SchedulerPhaseConfig`
- `SchedulerPhaseLike`
- `create_scheduler`
- `make_scheduler_config`
- `LLMOptimizationConfig`
- `create_llm_optimizer`
- `create_llm_lightning_optimizers`

`torch.optim.Muon` 是依赖边界之一，因此 `anytrain` 要求 `torch>=2.12`。

## AdamW

`create_adamw_optimizer(module, config)` 会拆分 decay / no-decay 参数组。默认 `decay_policy=AdamWDecayPolicy.STANDARD`，也就是常见 AdamW 规则：

- decay：非 embedding / norm / 显式排除 module 中，名称为 `weight` 且维度不小于 2 的参数。
- no-decay：bias、norm、embedding，以及用户显式排除的 module / module type。

LLM 预设会显式使用 `decay_policy=AdamWDecayPolicy.MUON_ELIGIBLE`，让 AdamW decay 规则和 Muon 参数语义一致：只 decay 适合放进 Muon 的 hidden 2D weight。为了方便 Hydra/YAML 配置，`decay_policy` 也接受字符串 `"standard"` 和 `"muon_eligible"`。

`optim` 不按名字猜 output head。需要排除 head 时，显式传 module 对象：

```python
optimizer = create_adamw_optimizer(
    model,
    AdamWConfig(lr=3e-4),
    excluded_modules=(model.lm_head,),
)
```

## Muon

`split_muon_params(module)` 默认将 hidden 2D weight 放入 Muon，其余参数留给 AdamW。默认排除：

- embedding。
- norm。
- bias 和非 `weight` 参数。
- 非 2D weight。
- `excluded_modules` 传入的 module 及其子 module。
- `excluded_module_types` 匹配的 module。

因此 head 不再因为名字包含 `head` 自动排除；如果要让 head 走 AdamW，传入具体 module：

```python
optimizer = create_muon_adamw_optimizer(
    model,
    muon=MuonConfig(lr=3e-4),
    adamw=AdamWConfig(lr=3e-4),
    excluded_modules=(model.lm_head,),
)
```

`create_muon_adamw_optimizer(module, muon=..., adamw=...)` 返回 `CompositeOptimizer`：

- `"muon"` 子 optimizer 处理 Muon 参数。
- `"adamw"` 子 optimizer 处理其余参数，并全部使用 `weight_decay=0.0`。

`MuonConfig.adjust_lr_fn` 默认是 `MuonAdjustLRFn.MATCH_RMS_ADAMW`，用于和 AdamW 常用 lr/wd 配置对齐。PyTorch 原生 `torch.optim.Muon` 默认不设置 `adjust_lr_fn`；这里显式改成 LLM 训练更常用的对齐口径。为了方便 Hydra/YAML 配置，`adjust_lr_fn` 也接受字符串 `"original"` 和 `"match_rms_adamw"`。

## Scheduler

`scheduler.py` 提供组合式 step-level scheduler。每个 `SchedulerPhaseConfig` 表示一个连续 phase：

```python
SchedulerPhaseConfig(
    shape="linear",          # 也可以用 CurveShape.LINEAR
    duration_steps=1000,    # -1 表示无限尾段，只能用于最后一个 constant phase
    start_lr_ratio=0.0,     # None 表示接上一个 phase 的 end_lr_ratio
    end_lr_ratio=1.0,
)
```

`shape` 会规范化为 `CurveShape`，可选值是 `CONSTANT`、`LINEAR`、`COSINE`。

多个 phase 通过 `SchedulerConfig(phases=(...))` 串起来。旧的 warmup + cosine 可以写成：

```python
SchedulerConfig(
    phases=(
        SchedulerPhaseConfig("linear", duration_steps=1000, start_lr_ratio=0.0, end_lr_ratio=1.0),
        SchedulerPhaseConfig("cosine", duration_steps=99000, end_lr_ratio=0.1),
    )
)
```

WSD 可以写成：

```python
SchedulerConfig(
    phases=(
        SchedulerPhaseConfig("linear", duration_steps=1000, start_lr_ratio=0.0, end_lr_ratio=1.0),
        SchedulerPhaseConfig("constant", duration_steps=90000, end_lr_ratio=1.0),
        SchedulerPhaseConfig("cosine", duration_steps=10000, end_lr_ratio=0.1),
    )
)
```

`create_scheduler(optimizer, config)` 返回 `torch.optim.lr_scheduler.LambdaLR`。所有 phase 都由自己的 `duration_steps` 定义长度；有限 phase 全部结束后会保持最后一个 `end_lr_ratio`。

`duration_steps=-1` 是无限尾段。它会让后续 phase 不可达，因此只能出现在最后一个 phase，并且只支持 `constant`。

如果只需要默认 ratio，可以用简便接口：

```python
make_scheduler_config(
    ("linear", 1000),
    ("constant", 90000),
    ("cosine", 10000),
)
```

## CompositeOptimizer

`CompositeOptimizer` 把多个真实 optimizer 暴露成一个 `torch.optim.Optimizer`：

- `param_groups` 引用子 optimizer 的真实 param group，因此 scheduler 修改 lr 时会同步影响子 optimizer。
- `step()` 依次调用子 optimizer。
- `zero_grad()` 依次清空子 optimizer。
- `state_dict()` / `load_state_dict()` 按 optimizer 名称保存和恢复。
- 子 optimizer 不能共享参数，否则会在构造时抛错，避免同一参数被 step 两次。
- 构造后不支持直接对 composite 调用 `add_param_group()`；新参数组必须先加到真实子 optimizer 上。

第一版 closure 只执行一次，并把结果作为 composite `step()` 返回值；子 optimizer 的 `step()` 不再接收 closure，避免重复执行 forward/backward。

## LLM Helper

`LLMOptimizationConfig` 用于按显式的 optimizer 配置创建 LLM 常用 optimizer。optimizer 类型不再单独配置，而是由 `optimizer_config` 的类型决定：

- `AdamWConfig`: 创建 AdamW。
- `MuonAdamWConfig`: 创建 Muon + AdamW 组合。

`create_llm_optimizer()` 根据配置调用 AdamW 或 Muon+AdamW helper。

LLM preset 通过字符串指定，内部再规范化为枚举：

```python
adamw_config = LLMOptimizationConfig.from_preset("sft", optimizer="adamw")
muon_config = LLMOptimizationConfig.from_preset(
    "pretrain",
    optimizer="muon",
    scheduler=[("linear", 1000), ("cosine", 10000)],
)
```

如果要排除 output head，可以在下游 LightningModule 拿到模型后直接构造配置：

```python
config = LLMOptimizationConfig.from_preset(
    "pretrain",
    optimizer="muon",
    excluded_modules=(model.lm_head,),
)
```

可用 `preset` 是 `"pretrain"`、`"cpt"`、`"sft"`；可用 `optimizer` 是 `"adamw"`、`"muon"`。

`from_preset()` 的 `scheduler` 是轻量 phase 列表：

- 默认是无限 `constant` phase，也就是保持 optimizer 原始 lr。
- 每项是 `(shape, duration_steps)`，例如 `[("linear", 1000), ("cosine", 10000)]`。
- 完整 scheduler 配置仍可通过直接构造 `LLMOptimizationConfig` 使用。

`create_llm_lightning_optimizers()` 返回 Lightning `configure_optimizers()` 可直接返回的 dict：

```python
return create_llm_lightning_optimizers(
    self.model,
    self.optim_config,
)
```

## 边界

`optim` 不做：

- 不解释 batch schema。
- 不替下游选择模型结构。
- 不作为 Hydra 顶层硬字段自动注入。
- 不提供要求继承的 LightningModule 基类。
- 不隐藏多 optimizer 或 scheduler 的返回结构；下游仍可直接按 Lightning 原生写法返回。
