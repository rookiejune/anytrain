# TODO

先做训练期 evaluator 的 core 形状，再做 optional 领域 evaluator。

## Core

1. 已完成第一版 `EvaluatorABC`：有状态 evaluator 抽象基类，子类实现 `evaluate()`。
2. 已完成第一版 `MetricValue`：只允许 Python `float` 或 0 维 `torch.Tensor`。
3. 已完成第一版 `EvaluatorGroup`：组合多个 evaluator，统一 key 校验。
4. `EvaluatorABC` 基类统一提供 `update/compute/reset`、metric 校验和 `dict[str, list[MetricValue]]` 状态。
5. 不解释 batch schema；只处理 prediction/target 或用户传入的显式参数。

## Next Core

1. 根据下游真实使用情况决定是否增加常用 stateful evaluator，例如 MAE/MSE/accuracy。
2. 如果引入 torchmetrics-backed 通用指标，先放 optional 子模块，不放 core。

## Optional

1. `evaluator.metrics`：torchmetrics-backed 通用分类/回归 evaluator。
2. `evaluator.audio`：codec/speech/audio quality evaluator。
3. `evaluator.text`：文本生成/分类 evaluator。
4. `evaluator.speech`：ASR、UTMOS 和 speech-to-text reference metric evaluator。

## Speech/Text Evaluator 设计计划

目标：先补 optional evaluator 的最小可用闭环，不改变 core `EvaluatorABC` / `EvaluatorGroup` 抽象，不解释 batch schema，只处理用户显式传入的 audio/text 参数。

### 包结构

新增 optional 子模块：

```text
src/anytrain/evaluator/
  speech/
    __init__.py
    asr.py
    utmos.py
  text/
    __init__.py
    evaluator.py
    normalization.py
    scores.py
```

依赖策略：

- core `anytrain.evaluator` 不 import speech/text 子模块，避免把领域依赖带入默认安装。
- 缺 optional 依赖时在具体 evaluator 初始化时报清晰错误，提示安装对应 extra。
- `pyproject.toml` 后续增加 extras：
  - `speech`：Whisper ASR、UTMOS 所需依赖。
  - `text`：BLEU、WER、chrF 所需文本指标依赖；优先考虑 `sacrebleu`/`jiwer` 这类成熟实现。
  - `evaluator`：聚合 `speech` 和 `text`。

### `evaluator.speech`

第一版提供两个能力：

1. `UTMOSEvaluator`
   - 输入：`prediction_audio` 和采样率，支持单条 waveform 或 batch waveform。
   - 输出：`{"utmos": float | 0-d tensor}`，batch 时默认对 batch 求均值。
   - 模型加载放在 evaluator 初始化阶段；device 显式传入或跟随输入 tensor。
   - 不在 core 内置 checkpoint 下载逻辑；优先使用第三方包的官方加载方式，缓存位置遵循其默认规则或由用户显式配置。
   - 缺依赖、采样率不合法、输入 shape 不支持时直接抛错，不做静默兼容。

2. `WhisperASREvaluator`
   - 输入：`prediction_audio`、采样率，以及可选 `target_text`。
   - 行为：先用 Whisper 转写 audio；如果提供 `target_text`，调用 text evaluator 计算 BLEU/WER/chrF；如果不提供，只返回转写文本不适合放进 `MetricDict`，因此第一版 evaluator 只暴露带 reference 的 metric 路径。
   - 输出：至少 `{"bleu": float, "wer": float, "chrf": float}`；后续可扩展 `match_rate`、`normalized_wer`。
   - Whisper 模型名、语言、task、device、解码参数显式配置。
   - ASR 转写文本可通过单独 helper 方法返回，例如 `transcribe(...)`，但 `evaluate()` 只返回数值 metric，保持 `MetricDict` 约束。

### `evaluator.text`

第一版面向两个 text 的比较，不做文本分类 task schema：

- `TextComparisonEvaluator`
  - 输入：`prediction_text: str | Sequence[str]`，`target_text: str | Sequence[str]`。
  - 输出：默认包含 `bleu`、`wer`、`chrf`。
  - batch 输入时逐条计算后求均值；长度不一致直接抛错。
  - 默认 normalization 做最小处理：strip、统一空白、可选 lowercase。
  - normalization 配置显式化，例如 `lowercase: bool = True`、`collapse_whitespace: bool = True`、`strip: bool = True`。
  - 第一版优先用成熟库实现 BLEU/chrF/WER，避免手写指标和主流工具口径不一致；如果缺依赖，在初始化时报清晰错误。

文本指标定义：

- `bleu`：corpus BLEU，默认使用 normalized prediction/reference 后的整批文本。
- `wer`：word error rate，以 normalized whitespace token 为单位。
- `chrf`：character n-gram F-score，默认使用 chrF2 口径。

空 reference 处理需要显式定义：

- prediction 和 target 都为空时，WER 记为 `0.0`；BLEU/chrF 按依赖库结果返回或显式定义为 `0.0`，实现前固定测试口径。
- target 为空但 prediction 非空时，WER 记为 `1.0`；BLEU/chrF 按依赖库结果返回或显式定义为 `0.0`，实现前固定测试口径。

### 实现步骤

1. Done: 实现 `evaluator.text` 和测试，作为 speech ASR metric 的下层依赖。
2. Done: 增加 `evaluator.speech.asr`，封装 Whisper 转写，并复用 `TextComparisonEvaluator` 输出 BLEU/WER/chrF。
3. Done: 增加 `evaluator.speech.utmos`，封装 UTMOS 打分。
4. Done: 更新 `pyproject.toml` optional extras 和 `docs/modules/evaluator.md`。
5. Done: 增加测试：
   - text evaluator 的单条、batch、空 reference、长度不一致、normalization 和 BLEU/WER/chrF 口径。
   - speech evaluator 缺依赖时报错信息。
   - Whisper/UTMOS 的轻量 mock 测试；真实模型测试只放 manual 或标记为 slow，不进入默认 test。

### 后续

1. Todo: 接入真实 Whisper backend 时，在下游项目或 optional adapter 里负责模型加载、缓存和 device 放置。
2. Todo: 接入真实 UTMOS backend 时，明确 checkpoint 来源、采样率要求和 batch 推理行为。
3. Todo: 如果后续改用 `sacrebleu` / `jiwer`，保留当前 public key 和空 reference 策略。
