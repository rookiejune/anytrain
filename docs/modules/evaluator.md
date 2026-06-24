# `anytrain.evaluator` Design

## 定位

`anytrain.evaluator` 是训练期 metric/evaluator 组件层。它面向下游 LightningModule 的 `validation_step()`、`test_step()` 或需要监控的 `training_step()`，负责把 prediction/target 转成可记录指标。

## 当前结构

源码结构：

```text
src/anytrain/evaluator/
  __init__.py
  abc.py
  group.py
  speech/
    __init__.py
    asr.py
    utmos.py
  text/
    __init__.py
    evaluator.py
    scores.py
  todo.md
```

当前公开导出：

- `EvaluatorABC`
- `MetricValue`
- `MetricDict`
- `EvaluatorGroup`

optional 子模块导出：

- `anytrain.evaluator.text.TextComparisonEvaluator`
- `anytrain.evaluator.speech.SpeechEvaluator`
- `anytrain.evaluator.speech.WhisperASREvaluator`
- `anytrain.evaluator.speech.UTMOSEvaluator`

## 目标接口

core evaluator 的目标调用形态：

```python
metrics = evaluator(prediction, target)
```

返回值约定：

```python
dict[str, float | torch.Tensor]
```

第一版 core 组件：

- `EvaluatorABC`：继承 `torch.nn.Module` 的有状态 evaluator 抽象基类，子类实现 `evaluate()`；基类统一提供 `update()`、`compute()`、`reset()`。
- `MetricValue`：只允许 Python `float` 或 0 维 `torch.Tensor`。
- `MetricDict`：`dict[str, MetricValue]`，不再额外维护 metric map / mapping 类型。
- `EvaluatorGroup`：组合多个 `EvaluatorABC`，内部用 `nn.ModuleDict` 注册子 evaluator，并处理 key 校验。

`EvaluatorGroup` 接收 evaluator mapping：

```python
evaluator = EvaluatorGroup(
    {
        "error": ErrorEvaluator(),
        "quality": QualityEvaluator(),
    }
)
metrics = evaluator(prediction, target)
```

输出 key 会包含 evaluator 名称和 metric key，例如：

```text
error/mae
quality/snr
```

metric dict 必须是一层 `dict[str, MetricValue]`。metric key 和 evaluator name 不允许包含 separator，默认是 `/`。evaluator name 还需要满足 `nn.ModuleDict` 的原生命名规则。

evaluator 继承 `EvaluatorABC`，只实现单步 `evaluate()`：

```python
class RunningMAE(EvaluatorABC):
    def evaluate(self, prediction, target):
        return {"mae": (prediction - target).abs().mean()}
```

`EvaluatorABC.__call__()` 和 `EvaluatorGroup.forward()` 会调用 `evaluate()`，校验并返回合并后的 metric dict，不写入内部状态。

`EvaluatorABC.update()` 会在 `evaluate()` 后统一校验返回值并写入内部状态。校验规则：

- 返回值必须是非空 `dict`。
- key 必须是非空字符串。
- key 不能包含 `EvaluatorABC.metric_key_separator`，默认是 `/`。
- value 必须是 Python `float` 或 0 维 `torch.Tensor`。
- Python `int`、`bool` 和非 0 维 tensor 会直接抛错。

需要 epoch / running metric 时显式使用状态生命周期：

```python
evaluator.update(prediction, target)
metrics = evaluator.compute()
evaluator.reset()
```

`EvaluatorABC` 内部持有 `dict[str, list[MetricValue]]`，`compute()` 默认对每个 key 求均值，`reset()` 清空状态。`EvaluatorGroup` 只接受 `EvaluatorABC` 实例，不再支持 stateless callable evaluator。

## 依赖分层

core evaluator 只依赖默认依赖，提供通用 ABC 和轻量组合器。

optional 子模块：

- `anytrain.evaluator.metrics`：torchmetrics-backed 通用分类/回归指标。
- `anytrain.evaluator.audio`：codec、speech、audio quality evaluator。
- `anytrain.evaluator.text`：文本生成/分类 evaluator。
- `anytrain.evaluator.speech`：ASR、UTMOS 和 speech-to-text reference metric evaluator。

optional 子模块缺依赖时应抛出清晰错误，提示安装对应 extra。

## Optional Text/Speech Evaluator

`anytrain.evaluator.text` 提供 `TextComparisonEvaluator`，比较 prediction/reference text，并返回：

- `bleu`：0-100 区间 sacreBLEU smoothed effective-order BLEU。
- `wer`：jiwer word error rate。
- `chrf`：0-100 区间 sacreBLEU chrF。

默认文本归一化会 strip、collapse whitespace、remove punctuation；`lowercase` 仍需显式打开。
需要安装 `anytrain[text]` 或 `anytrain[speech]` 使用默认 text evaluator。
缺少 `sacrebleu` 或 `jiwer` 时会显式 warning 并使用 `text._fallback` 的轻量实现。

`anytrain.evaluator.speech` 提供：

- `SpeechEvaluator`：组合 `WhisperASREvaluator` 和 `UTMOSEvaluator`，一次返回 `bleu`、`wer`、`chrf`、`utmos`。
- `WhisperASREvaluator`：默认用 `openai-whisper` 的 `large-v3` 转写 audio；通过 allowlisted `model_name`、`device`、`download_root`、`load_options` 选择官方 Whisper 配置，不接受本地 checkpoint path，也不开放任意 ASR backend/model 注入；再复用 text evaluator 输出 `bleu`、`wer`、`chrf`。
- `UTMOSEvaluator`：默认用 `torch.hub` 加载 `tarepan/SpeechMOS` 的 `utmos22_strong`，也可显式注入 UTMOS backend；输出 `utmos`。

`anytrain.evaluator.speech` 顶层只导出 evaluator。ASR 的 Whisper loader 是私有实现；
UTMOS backend 细节从 `anytrain.evaluator.speech.utmos` 显式导入。

默认 backend 都是延迟加载：构造 evaluator 不会下载或加载模型，第一次 `evaluate()` /
`transcribe()` 才会访问模型依赖。需要真实 speech evaluator 时安装 `anytrain[speech]`。
需要控制 checkpoint、cache 或 device 时，优先通过 evaluator 的显式配置项传入。
默认 torch backend 会在调用前将模型 `requires_grad_(False)`、`eval()`，并在
`torch.inference_mode()` 内执行模型前向或转写。

## 边界

`evaluator` 不做：

- 不解释 batch schema。
- 不决定 validation/test step 的运行方式。
- 不直接操作 Trainer。
- 不把领域 metric 放进 core import。
- 不隐藏第三方 metric backend 的状态生命周期。

## 测试策略

当前覆盖：

- 单个 evaluator 返回 dict。
- 多 evaluator 组合和 key 校验。
- `EvaluatorABC.update()` 后的通用校验和 detach 行为。
- epoch metric 的 `update/compute/reset` 生命周期。
- 错误返回类型、空 metrics、非 0 维 Tensor、Python 非 float 标量和非 ABC evaluator。

后续实现 optional evaluator 后，需要补充 optional 缺依赖时的错误信息。

当前 optional 覆盖：

- text evaluator 的单条、batch、normalization、空 reference、长度不一致和 metric key。
- speech evaluator 的默认 backend 延迟构造、ASR 配置项、ASR reference 校验、text metric 透传、UTMOS batch mean 和真实 backend 输出规范化。
