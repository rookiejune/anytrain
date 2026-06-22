# TODO

framework 是 optional/experimental 层，不进入 core。

1. 先保留目录边界，不把 flow matching / MAE / GAN helper 放进默认 import。
2. 每个 framework 子模块需要明确依赖 extra 和最小测试。
3. 只有跨项目复用明确后再从下游项目迁入。
4. framework 不替用户写完整 `pl_module`，只提供训练逻辑可复用组件。
