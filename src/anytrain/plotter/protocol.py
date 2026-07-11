from __future__ import annotations

from typing import Protocol, TypeVar

StateT = TypeVar("StateT", contravariant=True)
OutputT = TypeVar("OutputT", covariant=True)


class Plotter(Protocol[StateT, OutputT]):
    def __call__(self, state: StateT) -> OutputT:
        raise NotImplementedError
