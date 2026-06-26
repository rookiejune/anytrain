# TODO

## Core

1. `TaskLoss`：plain-config 友好的组合容器，不绑定具体任务语义。
2. 更多 balancer 策略，例如 deviation。

## Optional

1. GAN 相关训练目标已迁到 `framework.gan`；`loss` 下不再保留 GAN 子模块。
2. `loss.text` / `loss.speech`：只有在依赖和复用需求明确后再加。
