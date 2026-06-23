# Flow Matching Design

## 目标

`anytrain.framework.flow_matching` 作为 Facebook `flow_matching` 的轻量再封装，提供可组合进下游 `LightningModule` 的训练目标和采样 helper。

核心目标：

- 复用 Facebook `flow_matching` 的 path、loss 和 solver，不重写概率路径数学。
- 保留 deepaudio 中 `ContinuousFlowMatching` / `DiscreteFlowMatching` 的易用入口。
- 把 source distribution、time sampler、path、objective、solver 拆成可替换组件。
- 不绑定 audio/text/codec 任务语义，不解释 batch schema，不生成完整 `LightningModule`。
- optional 依赖不进入 `anytrain` 默认 import。

## 非目标

第一版不做：

- 完整 diffusion/flow 模型 zoo。
- 下游训练入口、配置系统、data module 或 task step。
- audio codec、text encoder、condition encoder 等领域组件。
- 自定义 probability path 数学实现，除非只是包一层已有 Facebook path。
- 和 deepaudio API 的逐字段兼容。

## deepaudio 参考

参考源码：

- `deepaudio/src/deepaudio/framework/flow_matching/abc.py`
- `deepaudio/src/deepaudio/framework/flow_matching/continuous.py`
- `deepaudio/src/deepaudio/framework/flow_matching/discrete.py`
- `deepaudio/src/deepaudio/task/sound_flow/pl_module.py`

可保留的思路：

- 连续空间默认使用 `CondOTProbPath`，目标是预测 velocity，loss 为 MSE。
- 离散空间默认使用 `MixtureDiscreteProbPath` 和 `MixturePathGeneralizedKL`。
- source distribution 支持连续的 gaussian/uniform、离散的 uniform/mask。
- sampling 通过 Facebook solver 完成，而不是自己写积分器。

需要调整的地方：

- deepaudio 的 helper 同时持有 path、source、loss、solver，并直接调用 task model；`anytrain` 应拆成可组合组件。
- deepaudio 依赖 `deepaudio.protocol.model.FlowModel`；`anytrain` 应只依赖 `torch.nn.Module` 或轻量 protocol。
- 连续和离散模型调用方式不统一；`anytrain` 第一版统一默认 `model(x_t, t, **extras)`，特殊模型用显式 adapter。
- 默认参数不要直接使用 `CondOTProbPath()` 这类对象实例，避免共享可变状态；构造函数用 `None` 后内部创建。
- `SoundFlow` 里的 codec/text 逻辑属于下游任务，不迁入。

## 目标包结构

推荐第一版放在：

```text
src/anytrain/framework/flow_matching/
  __init__.py
  _deps.py
  types.py
  source.py
  time.py
  objective.py
  sampler.py
  continuous.py
  discrete.py
```

职责：

- `_deps.py`：集中导入 Facebook `flow_matching`，缺依赖时抛清晰错误。
- `types.py`：轻量 protocol、dataclass 和 enum。
- `source.py`：source distribution 采样组件。
- `time.py`：训练和采样的时间网格组件。
- `objective.py`：连续 velocity matching、离散 generalized KL 等训练目标。
- `sampler.py`：对 Facebook solver 的薄封装，统一返回 final/intermediates。
- `continuous.py` / `discrete.py`：组合好的 preset，服务最小闭环。

公开导入第一版只从 `anytrain.framework.flow_matching` 导出稳定对象；`anytrain.framework.__init__` 不默认导入该子模块。

## 组合模型

第一版把 flow matching 拆成五类组件。

### Source

source 只负责从 target 或 shape 生成 `x_0`：

```python
class Source(Protocol):
    def sample_like(self, x_1: Tensor) -> Tensor: ...
```

建议实现：

- `GaussianSource`
- `UniformSource`
- `UniformTokenSource(vocab_size: int)`
- `MaskTokenSource(mask_id: int)`

连续 source 返回和 `x_1` 同 shape/device/dtype 的 tensor。离散 source 返回 `torch.long` token tensor；mask source 的 `mask_id` 必须显式传入，不静默假设等于 `vocab_size`。

### Time Sampler

time sampler 只负责给 batch 生成时间：

```python
class TimeSampler(Protocol):
    def sample(self, batch_size: int, device: torch.device) -> Tensor: ...
```

默认实现：

- 连续 preset 使用 `LogitNormalTimeSampler(mean=0.0, std=1.0, t_min=0.0, t_max=1.0)`，和采样的 `[0, 1]` 端点保持一致。
- 离散 preset 使用 `LogitNormalTimeSampler(mean=0.0, std=1.0, t_min=0.0, t_max=1.0 - 1e-3)`，和离散 Euler sampler 的 `[0, 1 - eps]` 端点保持一致。
- `LogitNormalTimeSampler` 经过 sigmoid 后不会实际采到端点；`t_min` / `t_max` 描述的是训练和采样约定的时间区间。不要默认把 `t_min` 设成非零值，否则推理时起点会变成未知的 `x_eps`。

### Objective

objective 负责把 `x_0`、`x_1`、`t` 和模型预测变成 scalar loss：

```python
loss = objective(model, x_1, x_0=None, **extras)
```

连续 preset：

- path：`CondOTProbPath`
- source：`GaussianSource`
- time：`LogitNormalTimeSampler`
- loss：`mse_loss(model(x_t, t, **extras), dx_t)`

离散 preset：

- path：`MixtureDiscreteProbPath(PolynomialConvexScheduler(n=2.0))`
- source：`UniformTokenSource` 或 `MaskTokenSource`
- time：`LogitNormalTimeSampler(t_min=0.0, t_max=1.0 - 1e-3)`
- loss：`MixturePathGeneralizedKL`
- `x_0` / `x_1` 必须是 `torch.long` token tensor，不做静默 dtype 转换。
- 模型输出 logits，objective 内部不自动做 sampling 或 argmax。

`objective` 返回必须是 scalar Tensor。额外日志不在第一版塞进返回值；后续如果需要，可加 `FlowLossOutput(loss, details)`，但不能影响 `backward()` 主路径。

`Objective` 是底层训练目标入口；`FlowMatcher` 是 preset 组合器入口。两者不要互相作为构造参数混用：需要完全自定义训练目标时，直接使用 objective；需要默认训练/采样组合时，使用 matcher。

### Model Caller

默认模型调用约定为：

```python
prediction = model(x_t, t, **extras)
```

如果下游模型需要 `model(x=..., t=...)`、condition 预处理或多个输入，可以显式传入：

```python
def call_model(model: nn.Module, x_t: Tensor, t: Tensor, extras: Mapping[str, object]) -> Tensor:
    return model(x=x_t, t=t, **extras)
```

这样保留组合自由度，但不把 batch schema 写进 `anytrain`。

### Solver

solver 负责 sampling：

```python
output = sampler.sample(model, x_0, **extras)
```

建议统一返回：

```python
@dataclass(eq=False)
class FlowSampleOutput:
    final: Tensor
    states: Tensor | None = None
    time_grid: Tensor | None = None
```

连续 sampler 默认封装 `ODESolver`，默认参数：

- `method="midpoint"`
- `nfe=20`
- `num_steps=10`
- `return_intermediates=True`

离散 sampler 默认封装 `MixtureDiscreteEulerSolver`，内部把 logits 转成概率：

```python
prob = model(x_t, t, **extras).softmax(dim=-1)
```

离散 sampler 的默认时间网格是 `[0, 1 - eps]`，和离散训练 objective 保持同一个有效时间区间。`vocab_size` 表示目标 token 空间大小；mask token 是否属于模型输入 embedding，由下游模型自己处理。

## 便利封装

为保留 deepaudio 的易用性，可以提供两个 preset：

```python
matcher = ContinuousFlowMatcher()
loss = matcher.loss(model, x_1, condition=condition)
x_0 = matcher.source.sample_like(x_1)
samples = matcher.sample(model, x_0, condition=condition)
```

```python
matcher = DiscreteFlowMatcher(vocab_size=1024, source=MaskTokenSource(mask_id=1024))
loss = matcher.loss(model, tokens, condition=condition)
x_0 = matcher.source.sample_like(tokens)
samples = matcher.sample(model, x_0, condition=condition)
```

这些类只是组合器，不继承或替代下游 `LightningModule`。用户也可以绕过 preset，单独组合 source、objective 和 sampler。

## 依赖策略

`flow_matching` 作为 optional framework 依赖，不进入默认安装。建议在 `pyproject.toml` 增加：

```toml
[project.optional-dependencies]
flow = ["flow_matching"]
```

依赖行为：

- `import anytrain` 不导入 `flow_matching`。
- `import anytrain.framework` 不导入 `flow_matching`。
- `import anytrain.framework.flow_matching` 如果缺依赖，抛出带安装提示的 `ImportError`。
- 单测要覆盖缺依赖错误路径，但不影响 core 测试。

## 第一版 API 草案

```python
from anytrain.framework.flow_matching import (
    ContinuousFlowMatcher,
    DiscreteFlowMatcher,
    FlowSampleOutput,
    GaussianSource,
    LogitNormalTimeSampler,
    MaskTokenSource,
    ODESampler,
    UniformTimeSampler,
    UniformTokenSource,
)
```

命名约定：

- 组合器用 `FlowMatcher`，表示训练目标和采样器的组合。
- 训练目标用 `Objective`，避免和 `torch.nn.Module.loss` 或 `anytrain.loss` 混淆。
- sampling helper 用 `Sampler`，不叫 `Solver`；`Solver` 特指 Facebook 原始 solver。
- `nfe` 可以保留，因为 flow matching 论文和 solver API 都常用这个名字。

## 实现计划

### P0: 设计冻结

- 增加本设计文档。
- 在 `docs/modules/framework.md` 和 docs index 加链接。
- 在 `src/anytrain/framework/todo.md` 记录分阶段任务。

### P1: 最小包结构和依赖边界

- 新增 `flow_matching` 包和 `_deps.py`。
- 新增 source/time/model caller 基础组件。
- 新增 import smoke：core import 不触发 optional 依赖。
- 新增缺依赖错误测试。

### P2: Continuous 最小闭环

- 实现 `ContinuousVelocityObjective`。
- 实现 `ODESampler`。
- 实现 `ContinuousFlowMatcher` preset。
- 增加 CPU toy model 测试：loss finite、backward finite、sample shape 正确。

### P3: Discrete 最小闭环

- 实现 `DiscreteGeneralizedKLObjective`。
- 实现 `DiscreteEulerSampler`。
- 实现 `DiscreteFlowMatcher` preset。
- 增加 CPU toy token model 测试：uniform/mask source、loss finite、sample shape 正确。

### P4: 文档示例

- 在 `docs/modules/framework.md` 补最小使用示例链接。
- 如有真实下游需求，再补 examples；examples 只展示普通 `LightningModule` 如何组合 helper，不提供完整 app。

## 验收标准

- 不需要 deepaudio import。
- 不依赖 audio/text/codec 包。
- `import anytrain` 和 `import anytrain.framework` 不触发 `flow_matching` import。
- 连续和离散目标都能在 CPU 上用 toy model 跑通 forward/backward。
- source/time/objective/sampler 能单独构造，也能通过 preset 组合。
- 缺 optional 依赖时错误信息包含安装方式。
- 文档示例和测试使用同一套调用约定。

## 风险和处理

### Facebook API 变化

把所有 Facebook `flow_matching` 导入集中到 `_deps.py`。如果上游 API 改名，修正面集中，不影响用户侧组合接口。

### 过度封装

第一版只封 source、time、objective、sampler 和 preset，不新增 registry、配置装配或 Lightning 基类。真实项目需要复杂配置时，由下游项目自行选择 Hydra/pydantic/普通 Python。

### 离散 mask token 语义

mask token 是输入状态的一部分，不一定属于目标 vocabulary。`MaskTokenSource` 显式接收 `mask_id`，`DiscreteFlowMatcher` 显式接收 `vocab_size`，避免把 `mask_id == vocab_size` 写成隐式规则。

### 额外日志需求

第一版 objective 只返回 scalar loss。需要 path sample、target velocity 或 per-token KL 时，后续再增加显式 debug output，不让训练主路径先复杂化。
