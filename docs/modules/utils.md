# `anytrain.utils` Design

## 定位

`anytrain.utils` 只放跨模块稳定复用的小工具。能放在具体模块旁边的 helper，应优先留在具体模块里。

## 当前状态

当前 `src/anytrain/utils/__init__.py` 为空，目录只保留边界和 `todo.md`。optimizer 相关 helper 已提升到顶层 `anytrain.optim`，不再放在 utils 下。

## 可接受内容

未来可以考虑放入：

- 通用 dict flatten / prefix helper。
- optional dependency 错误提示 helper。
- 多个模块都会使用、且没有更明确归属的小型 pure function。

这些 helper 需要有清晰类型提示和单测。

## 不接受内容

`utils` 不放：

- audio/io/checker/rpc/patchifier/overlap 等领域工具。
- 只被单个模块使用的私有 helper。
- 公共注册表抽象；优先在需要它的具体模块内放局部 mapping 或 helper。
- 静默吞错的兼容逻辑。
- 下游项目私有工具。

## 设计原则

- 新增 utility 前先确认是否能放到具体模块。
- API 命名要直接表达用途，不做过宽抽象。
- 缺少 optional 依赖时直接给出明确错误。
- 不为了复用一两行代码而引入公共工具。

## 测试策略

每个公开 utility 都需要独立单测。涉及 optional dependency 的 helper 需要覆盖依赖存在和缺失两条路径。
