# TODO

先做 core 接口，再做 optional 领域 loss。

## Core

1. [x] `LossABC`：返回 scalar tensor，可选返回 details。
2. [x] `LossGroup`：组合多个 loss，返回 `(total, details)`。
3. [x] `LossBalancerABC` / `MeanLossBalancer` / `FixedWeightLossBalancer` / `UncertaintyLossBalancer`：将多个 loss 合成为一个 scalar total。
4. `TaskLoss`：plain-config 友好的组合容器，不绑定具体任务语义。
5. [x] 支持子 loss 返回 `Tensor` 或 `(Tensor, dict)`，自动 flatten/prefix details。
6. 更多 balancer 策略，例如 deviation。

## Optional

1. [x] `loss.temporal` / `loss.spectral`：时域、谱域和 multi-scale mel/STFT loss。
2. [x] `loss.task`：codec preset 和任务级组合。
3. `loss.gan`：GAN loss 协议和常见实现；和 manual optimization helper 的边界要单独设计。
4. `loss.text` / `loss.speech`：只有在依赖和复用需求明确后再加。
