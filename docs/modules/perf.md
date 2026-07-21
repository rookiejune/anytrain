# anytrain.perf

## 定位

`anytrain.perf` 提供训练效率观测的通用计算组件。它定义 MFU（Model FLOPs Utilization）的统一口径，并提供参数量统计、代表性 forward FLOPs profiling、硬件峰值算力推断和 Lightning callback。

核心指标：

```text
MFU = model_flops_per_step / step_time / hardware_peak_flops
```

MFU 数值越高越好。`anytrain` 不额外记录反向含义的训练效率指标。

## 边界

`anytrain.perf` 负责：

- 统计模型参数量。
- 用 PyTorch profiler 在下游提供的代表性输入上估算 forward FLOPs。
- 提供 `training_flops_from_forward()`，显式把 forward FLOPs 转成训练 step FLOPs。
- 根据 GPU 型号和 compute dtype 查内置硬件表，得到硬件峰值算力。
- 在 Lightning 训练中记录 step time 和 MFU。

`anytrain.perf` 不负责：

- 替下游选择代表性输入。
- 猜测 batch schema。
- 猜测 tokens、frames、valid mask count 等任务数据量。
- 接管 Hydra、Trainer 或训练入口装配。

下游 job 可以显式覆盖 `hardware_peak_flops`。覆盖值会在日志元数据中标记为 `override`。

## 用法

```python
import torch
from anytrain.lightning import PerformanceCallback
from anytrain.perf import profile_forward_flops, training_flops_from_forward

forward_flops = profile_forward_flops(model, args=(example_batch,))
model_flops_per_step = training_flops_from_forward(forward_flops)

callback = PerformanceCallback(
    model_flops_per_step=model_flops_per_step,
    compute_dtype=torch.bfloat16,
)
```

如果模型包含 PyTorch profiler 尚不支持 FLOPs 的算子，应由下游直接传入 `model_flops_per_step`。
