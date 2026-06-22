# `anytrain.registry` Design

## 定位

`anytrain.registry` 提供轻量、类型友好的 registry。它用于显式注册和创建对象，不替代 Hydra，也不作为项目级模型 zoo 的隐藏入口。

## 当前实现

公开类：

```python
Registry[K, V]
```

主要方法：

- `register(key, value=None, *, replace=False)`：注册值，也可作为 decorator 使用。
- `get(key)`：按 key 获取值。
- `create(key, *args, **kwargs)`：获取 callable 后创建对象。
- `items()`、`keys()`、`values()`：返回底层 mapping view。
- `as_dict()`：返回浅拷贝。

同时实现：

- `__contains__`
- `__iter__`
- `__len__`

## 使用方式

直接注册：

```python
registry = Registry[str, type]()
registry.register("linear", torch.nn.Linear)
module = registry.create("linear", 4, 1)
```

decorator 注册：

```python
registry = Registry[str, type]()

@registry.register("custom")
class CustomModule:
    ...
```

## 错误策略

- 重复 key 默认抛出 `KeyError`。
- 需要覆盖时显式传入 `replace=True`。
- 未知 key 抛出 `KeyError`，错误信息包含 available keys。
- `create()` 遇到不可调用的注册值时抛出 `TypeError`。

## 边界

`registry` 不做：

- 不扫描文件系统。
- 不自动 import 插件。
- 不处理 Hydra config。
- 不承担模型 zoo、checkpoint 下载或版本选择。
- 不静默覆盖已存在 key。

## 测试策略

当前覆盖：

- 注册并创建 callable。
- 重复 key 抛错。

后续如果 registry 用于更多组件，应补充 decorator 注册、`replace=True`、未知 key 错误信息和不可调用值的 `create()` 错误测试。

