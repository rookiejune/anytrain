# `anytrain.plotter` Design

## 定位

`anytrain.plotter` 是训练期可视化组件层。它负责把 tensor、metric 或中间状态转换为 figure 或 logger 可记录对象；实际写入 logger 的动作由下游 LightningModule 完成。

## 当前状态

当前提供最小公开 API：

- `Plotter`：描述 `plotter(state) -> output` 调用形态的轻量 protocol。
- `TensorImagePlotter`：把单张 CPU/GPU tensor 图像转换成 matplotlib figure。

`plotter` 属于 optional general 组件，默认 import 不会加载 matplotlib、plotly 或 seaborn；只有调用具体 matplotlib-backed plotter 时才需要 `plot` extra。

## 接口

plotter 的调用形态：

```python
figure = plotter(state)
```

设计约定：

- plotter 只生成可记录对象，不直接调用 Trainer。
- figure 类型由具体子模块决定，可以是 matplotlib figure、plotly figure 或图像 tensor。
- 记录频率、global step、logger backend 由 Lightning 层或下游项目决定。
- audio/image/MoE 等具体 plotter 放在 optional 子模块。

## 依赖策略

`plot` extra 当前依赖：

- `matplotlib`
- `plotly`
- `seaborn`

core import 不依赖这些库。缺少 plot extra 时，相关子模块应抛出明确错误，例如提示安装 `matplotlib`、`plotly` 和 `seaborn`（`python -m pip install matplotlib plotly seaborn`）。

## 边界

`plotter` 不做：

- 不解释 batch schema。
- 不决定训练 step 中何时画图。
- 不直接操作 logger 或 Trainer。
- 不把 audio/image 等领域可视化放进 core。
- 不在默认 import 路径中加载重可视化依赖。

## 测试策略

至少需要覆盖：

- 缺少 optional 依赖时错误信息清晰。
- plotter 在 CPU tensor 输入下可生成对象。
- 输出对象能被目标 logger backend 接受。
- 图像维度、颜色映射和 batch 选择逻辑不会随输入 batch size 产生异常。
