# TODO

framework 是 optional/experimental 层，不进入 core。

1. 每个新增 framework 子模块需要明确依赖 extra 和最小测试。
2. masked autoencoder 等训练范式只有跨项目复用明确后再迁入。
3. `framework.gan` 后续根据真实复用需求补 manual optimization helper 和 audio discriminator。
4. framework 不替用户写完整 `pl_module`，只提供训练逻辑可复用组件。
