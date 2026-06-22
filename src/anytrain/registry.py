from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import TypeVar

K = TypeVar("K")
V = TypeVar("V")


class Registry[K, V]:
    def __init__(self, initial: Mapping[K, V] | None = None):
        self._items: dict[K, V] = dict(initial or {})

    def register(self, key: K, value: V | None = None, *, replace: bool = False):
        if value is None:
            def decorator(candidate: V) -> V:
                self.register(key, candidate, replace=replace)
                return candidate

            return decorator

        if not replace and key in self._items:
            raise KeyError(f"{key!r} is already registered.")
        self._items[key] = value
        return value

    def get(self, key: K) -> V:
        try:
            return self._items[key]
        except KeyError as exc:
            available = ", ".join(str(item) for item in self._items)
            raise KeyError(f"Unknown registry key {key!r}. Available: {available}") from exc

    def create(self, key: K, *args, **kwargs):
        factory = self.get(key)
        if not callable(factory):
            raise TypeError(f"Registered value for {key!r} is not callable.")
        return factory(*args, **kwargs)

    def items(self):
        return self._items.items()

    def keys(self):
        return self._items.keys()

    def values(self):
        return self._items.values()

    def as_dict(self) -> dict[K, V]:
        return dict(self._items)

    def __contains__(self, key: object) -> bool:
        return key in self._items

    def __iter__(self) -> Iterator[K]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)
