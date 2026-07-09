from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from anytrain._compat import Self, strict_zip

from .corpus import _corpus_factory, _private_use_char, _scan_base_corpus, _text_corpus
from .frame import _normalize_base_id
from .interop import _core_state_from_tokenizer, _train_tokenizers_bpe
from .stats import Merge
from .types import BaseCorpus, BaseCorpusFactory, RepeatInterleaveOutput


class _CoreBPE:
    def __init__(
        self,
        tokens: Mapping[int, Sequence[int]],
        merges: Sequence[Merge | tuple[int, int, int]] = (),
    ) -> None:
        if not tokens:
            raise ValueError("tokens must not be empty")

        self.tokens = {
            int(token_id): tuple(_normalize_base_id(base_id) for base_id in base_ids)
            for token_id, base_ids in tokens.items()
        }
        for token_id, base_ids in self.tokens.items():
            if token_id < 0:
                raise ValueError("token ids must be non-negative")
            if not base_ids:
                raise ValueError(f"token {token_id} must contain at least one frame")

        parsed: list[Merge] = []
        for merge in merges:
            if isinstance(merge, Merge):
                parsed.append(merge)
            else:
                left, right, token_id = merge
                parsed.append(Merge(left=left, right=right, token_id=token_id))
        self.merges = tuple(parsed)

        self.base_to_id = self._build_base_to_id(self.tokens)

    @property
    def vocab_size(self) -> int:
        return max(self.tokens) + 1

    @classmethod
    def train(
        cls,
        corpus: BaseCorpus | BaseCorpusFactory,
        *,
        vocab_size: int = 30_000,
        min_frequency: int = 0,
        show_progress: bool = True,
        max_token_length: int | None = None,
    ) -> Self:
        corpus_factory = _corpus_factory(corpus)
        base, num_training_sequences = _scan_base_corpus(
            corpus_factory,
            show_progress=show_progress,
        )
        base_tokens = {
            base_id: _private_use_char(index) for index, base_id in enumerate(sorted(base))
        }
        tokenizer = _train_tokenizers_bpe(
            _text_corpus(corpus_factory, base_tokens),
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            show_progress=show_progress,
            max_token_length=max_token_length,
            length=num_training_sequences,
        )
        tokens, merges = _core_state_from_tokenizer(tokenizer, base_tokens)
        return cls(tokens, merges)

    def base_ids(self, token_id: int) -> tuple[int, ...]:
        if token_id in self.tokens:
            return self.tokens[token_id]
        raise KeyError(f"unknown token_id: {token_id}")

    def _token_lengths(self) -> dict[int, int]:
        return {token_id: len(base_ids) for token_id, base_ids in self.tokens.items()}

    def decode(self, token_ids: Sequence[int]) -> list[int]:
        base_ids, _ = self.decode_with_counts(token_ids)
        return base_ids

    def decode_with_counts(self, token_ids: Sequence[int]) -> tuple[list[int], list[int]]:
        if not token_ids:
            raise ValueError("token_ids must not be empty")

        base_ids: list[int] = []
        counts: list[int] = []
        for token_id in token_ids:
            ids = self.base_ids(token_id)
            base_ids.extend(ids)
            counts.append(len(ids))
        return base_ids, counts

    def repeat_interleave(
        self,
        x: torch.Tensor,
        token_ids: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        dim: int = -2,
    ) -> RepeatInterleaveOutput:
        if x.dim() == 0:
            raise ValueError("x must have at least one dimension")
        self._validate_token_ids(token_ids)

        dim = self._normalize_dim(dim, x.dim())
        if token_ids.dim() == 1:
            if mask is not None:
                raise ValueError("mask is only supported for 2D token_ids")
            return self._repeat_interleave_1d(x, token_ids, dim=dim)

        if token_ids.dim() == 2:
            return self._repeat_interleave_2d(x, token_ids, mask, dim=dim)

        raise ValueError("token_ids must be a 1D or 2D tensor")

    def _repeat_interleave_1d(
        self,
        x: torch.Tensor,
        token_ids: torch.Tensor,
        *,
        dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x.size(dim) != token_ids.numel():
            raise ValueError("x and token_ids must align on the sequence dimension")

        base_ids, counts = self.decode_with_counts(
            [int(token_id) for token_id in token_ids.tolist()],
        )
        repeats = torch.tensor(counts, dtype=torch.long, device=x.device)
        expanded_x = torch.repeat_interleave(x, repeats, dim=dim)
        expanded_ids = torch.tensor(base_ids, dtype=token_ids.dtype, device=token_ids.device)
        return expanded_x, expanded_ids

    def _repeat_interleave_2d(
        self,
        x: torch.Tensor,
        token_ids: torch.Tensor,
        mask: torch.Tensor | None,
        *,
        dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if dim == 0:
            raise ValueError("batched repeat_interleave expects a non-batch sequence dim")

        batch_size, seq_len = token_ids.shape
        if x.size(0) != batch_size:
            raise ValueError("x and token_ids must align on the batch dimension")
        if x.size(dim) != seq_len:
            raise ValueError("x and token_ids must align on the sequence dimension")

        token_mask = self._normalize_mask(mask, token_ids)
        pad_token_id = self._pad_token_id(token_ids, token_mask)

        row_dim = dim - 1
        expanded_rows: list[torch.Tensor] = []
        expanded_id_rows: list[torch.Tensor] = []
        for row in range(batch_size):
            positions = token_mask[row].nonzero(as_tuple=False).flatten()
            row_x = x.select(0, row).index_select(row_dim, positions.to(device=x.device))
            row_token_ids = token_ids[row].index_select(0, positions)
            expanded_x, expanded_ids = self._repeat_interleave_1d(
                row_x,
                row_token_ids,
                dim=row_dim,
            )
            expanded_rows.append(expanded_x)
            expanded_id_rows.append(expanded_ids)

        row_lengths = [row.numel() for row in expanded_id_rows]
        max_len = max(row_lengths)
        if all(length == max_len for length in row_lengths):
            return (
                torch.stack(expanded_rows, dim=0),
                torch.stack(expanded_id_rows, dim=0),
                torch.ones(
                    (batch_size, max_len),
                    dtype=torch.bool,
                    device=token_ids.device,
                ),
            )

        if pad_token_id is None:
            raise ValueError("cannot infer padding id from token_ids and mask")
        pad_base_id = self._pad_base_id(pad_token_id)

        out_shape = list(x.shape)
        out_shape[dim] = max_len
        expanded_x = x.new_zeros(out_shape)
        expanded_ids = token_ids.new_full((batch_size, max_len), pad_base_id)
        expanded_mask = torch.zeros(
            (batch_size, max_len),
            dtype=torch.bool,
            device=token_ids.device,
        )

        for row, (row_x, row_ids) in enumerate(strict_zip(expanded_rows, expanded_id_rows)):
            length = row_ids.numel()
            row_target = expanded_x.select(0, row)
            slices = [slice(None)] * row_target.dim()
            slices[row_dim] = slice(0, length)
            row_target[tuple(slices)] = row_x
            expanded_ids[row, :length] = row_ids
            expanded_mask[row, :length] = True

        return expanded_x, expanded_ids, expanded_mask

    @staticmethod
    def _validate_token_ids(token_ids: torch.Tensor) -> None:
        if (
            token_ids.dtype == torch.bool
            or torch.is_floating_point(token_ids)
            or torch.is_complex(token_ids)
        ):
            raise TypeError("token_ids must contain integer ids")

    @staticmethod
    def _normalize_mask(mask: torch.Tensor | None, token_ids: torch.Tensor) -> torch.Tensor:
        if mask is None:
            return torch.ones_like(token_ids, dtype=torch.bool)
        if mask.shape != token_ids.shape:
            raise ValueError("mask must have the same shape as token_ids")
        if mask.device != token_ids.device:
            raise ValueError("mask and token_ids must be on the same device")
        if torch.is_floating_point(mask) or torch.is_complex(mask):
            raise TypeError("mask must contain boolean or integer values")
        if mask.dtype != torch.bool and not torch.all((mask == 0) | (mask == 1)):
            raise ValueError("integer mask values must be 0 or 1")
        return mask.to(dtype=torch.bool)

    @staticmethod
    def _pad_token_id(token_ids: torch.Tensor, mask: torch.Tensor) -> int | None:
        pad_values = token_ids.masked_select(~mask)
        if pad_values.numel() == 0:
            return None
        unique = torch.unique(pad_values)
        if unique.numel() != 1:
            raise ValueError("token_ids padding values must be identical")
        return int(unique.item())

    def _pad_base_id(self, token_id: int) -> int:
        base_ids = self.base_ids(token_id)
        if len(base_ids) != 1:
            raise ValueError("padding token_id must expand to exactly one frame")
        return base_ids[0]

    @staticmethod
    def _build_base_to_id(tokens: Mapping[int, tuple[int, ...]]) -> dict[int, int]:
        base_to_id: dict[int, int] = {}
        for token_id, base_ids in tokens.items():
            if len(base_ids) == 1:
                base_to_id[base_ids[0]] = token_id
        return base_to_id

    @staticmethod
    def _normalize_dim(dim: int, ndim: int) -> int:
        if dim < 0:
            dim += ndim
        if dim < 0 or dim >= ndim:
            raise ValueError("dim is out of range for x")
        return dim
