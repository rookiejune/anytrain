# `anytrain.types` Design

## 定位

`anytrain.types` 放置跨模块可复用的轻量基础类型。当前只包含自动命名枚举，不承载业务 schema 或训练配置类型。

## 当前实现

公开类：

```python
class AutoNameEnum(str, Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name.lower()
```

`AutoNameEnum` 用于配合 `enum.auto()` 生成小写字符串值：

```python
from enum import auto
from anytrain.types import AutoNameEnum

class Mode(AutoNameEnum):
    TRAIN = auto()

assert Mode.TRAIN.value == "train"
```

如果枚举成员显式指定字符串值，则使用显式值。

## 设计原则

- 只放没有领域语义的基础类型。
- 不放 batch schema、sample schema 或任务配置 schema。
- 不引入 torch、Lightning、Hydra 之外的额外依赖。
- 类型行为应简单透明，避免大量兼容逻辑。

## 边界

`types` 不做：

- 不定义训练数据结构。
- 不定义模型输出结构。
- 不定义 evaluator/loss 的复杂协议，除非这些协议已经在对应模块稳定。
- 不作为 `utils` 的混杂替代品。

## 测试策略

当前覆盖显式字符串值保留。后续应补充 `enum.auto()` 自动小写值的测试，确保 `AutoNameEnum` 的核心行为被直接验证。

