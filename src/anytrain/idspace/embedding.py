from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from .layout import Layout


class Embedding(nn.Module):
    def __init__(
        self,
        layout: Layout,
        *,
        adapters: Mapping[str, nn.Module] | None = None,
        **embeddings: nn.Embedding,
    ) -> None:
        super().__init__()
        self.layout = layout
        self.embeddings = _init_embeddings(layout, embeddings)
        self.adapters = _init_adapters(layout, adapters)

    @property
    def num_embeddings(self) -> int:
        return self.layout.vocab_size

    @property
    def vocab_size(self) -> int:
        return self.num_embeddings

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        output: torch.Tensor | None = None
        covered = torch.zeros(input_ids.shape, dtype=torch.bool, device=input_ids.device)

        for name, embed in self.embeddings.items():
            start, end = self.layout.blocks[name]
            mask = (input_ids >= start) & (input_ids < end)
            if not bool(mask.any()):
                continue

            local_ids = input_ids[mask] - start
            local_output = embed(local_ids)
            if name in self.adapters:
                local_output = self.adapters[name](local_output)
            if not isinstance(local_output, torch.Tensor):
                raise TypeError(f"embedding block {name!r} must return a tensor.")

            if output is None:
                output = local_output.new_empty((*input_ids.shape, local_output.shape[-1]))
            elif local_output.shape[-1] != output.shape[-1]:
                raise ValueError("all selected embedding blocks must produce the same output dim.")
            output[mask] = local_output
            covered |= mask

        if not bool(covered.all()):
            bad_id = int(input_ids[~covered].reshape(-1)[0].detach().cpu())
            raise ValueError(f"input_ids contains id outside space: {bad_id}.")
        if output is None:
            raise ValueError("input_ids must not be empty.")
        return output


def _init_embeddings(
    layout: Layout,
    embeddings: Mapping[str, nn.Embedding],
) -> nn.ModuleDict:
    expected = set(layout.block_names)
    actual = set(embeddings)
    missing = expected - actual
    if missing:
        raise ValueError(f"missing embeddings for id blocks: {sorted(missing)!r}.")
    unknown = actual - expected
    if unknown:
        raise ValueError(f"unknown embedding blocks: {sorted(unknown)!r}.")

    modules = nn.ModuleDict()
    for name in layout.block_names:
        embed = embeddings[name]
        _validate_weight_size(name, embed, layout.blocks[name])
        modules[name] = embed
    return modules


def _init_adapters(
    layout: Layout,
    adapters: Mapping[str, nn.Module] | None,
) -> nn.ModuleDict:
    if adapters is None:
        return nn.ModuleDict()
    unknown = set(adapters) - set(layout.block_names)
    if unknown:
        raise ValueError(f"unknown adapter blocks: {sorted(unknown)!r}.")

    modules = nn.ModuleDict()
    for name, adapter in adapters.items():
        if not isinstance(adapter, nn.Module):
            raise TypeError(f"adapters[{name!r}] must be an nn.Module.")
        modules[name] = adapter
    return modules


def _validate_weight_size(name: str, embed: nn.Embedding, block: tuple[int, int]) -> None:
    start, end = block
    expected = end - start
    if embed.num_embeddings != expected:
        raise ValueError(
            f"embeddings[{name!r}].num_embeddings must match id block size {expected}."
        )


__all__ = [
    "Embedding",
]
