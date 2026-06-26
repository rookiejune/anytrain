# TODO

## Core

1. 根据下游真实使用情况决定是否增加常用 stateful evaluator，例如 MAE/MSE/accuracy。
2. 如果引入 torchmetrics-backed 通用指标，先放 optional 子模块，不放 core。

## Optional

1. `evaluator.metrics`：torchmetrics-backed 通用分类/回归 evaluator。
2. `evaluator.audio`：codec/speech/audio quality evaluator。
3. Whisper backend 包装层：固定模型名、语言、device 和 decode options 配置入口。
4. UTMOS backend 包装层：明确 checkpoint 来源、cache、device、采样率要求和 batch 推理行为。
5. 如果后续改用 `sacrebleu` / `jiwer`，保留当前 public key 和空 reference 策略。
