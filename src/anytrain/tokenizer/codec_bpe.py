from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence, Sized
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, NotRequired, Self, TypedDict

import torch

if TYPE_CHECKING:
    from tokenizers import Tokenizer
    from tokenizers.models import BPE as TokenizersBPE
    from tqdm.std import tqdm as Tqdm

Unit = int | tuple[int, ...]
UnitInput = int | Sequence[int]
UnitState = int | list[int]

PRIVATE_USE_RANGES = (
    (0xE000, 0xF8FF),
    (0xF0000, 0xFFFFD),
    (0x100000, 0x10FFFD),
)


class _TokenizersBPEKwargs(TypedDict):
    vocab: dict[str, int]
    merges: list[tuple[str, str]]
    cache_capacity: NotRequired[int]
    dropout: NotRequired[float]
    unk_token: NotRequired[str]
    continuing_subword_prefix: NotRequired[str]
    end_of_word_suffix: NotRequired[str]
    fuse_unk: NotRequired[bool]
    byte_fallback: bool
    ignore_merges: bool


class CodecBPEState(TypedDict):
    tokens: dict[str, list[UnitState]]
    merges: list[dict[str, int]]
    strict: bool


# `tokenizers.models.BPE` is a Rust extension type and cannot be subclassed in Python.
@dataclass(frozen=True, slots=True)
class Merge:
    left: int
    right: int
    token_id: int


@dataclass(frozen=True, slots=True)
class CompressionStats:
    num_sequences: int
    original_tokens: int
    encoded_tokens: int
    mean_original_length: float
    mean_encoded_length: float
    compression_ratio: float
    compression_factor: float
    compression_gain: float


RepeatInterleaveOutput = tuple[torch.Tensor, torch.Tensor] | tuple[
    torch.Tensor, torch.Tensor, torch.Tensor
]


class _CoreBPE:
    def __init__(
        self,
        tokens: Mapping[int, Sequence[UnitInput]],
        merges: Sequence[Merge | tuple[int, int, int]] = (),
        *,
        strict: bool = True,
    ) -> None:
        if not tokens:
            raise ValueError("tokens must not be empty")

        self.strict = strict
        self.tokens = {
            int(token_id): tuple(_normalize_unit(unit) for unit in units)
            for token_id, units in tokens.items()
        }
        self.unit_dim = _unit_dim_from_token_units(self.tokens.values())
        for token_id, units in self.tokens.items():
            if token_id < 0:
                raise ValueError("token ids must be non-negative")
            if not units:
                raise ValueError(f"token {token_id} must contain at least one unit")

        parsed: list[Merge] = []
        for merge in merges:
            if isinstance(merge, Merge):
                parsed.append(merge)
            else:
                left, right, token_id = merge
                parsed.append(Merge(left=left, right=right, token_id=token_id))
        self.merges = tuple(parsed)

        self.unit_to_id = self._build_unit_to_id(self.tokens)

    @property
    def vocab_size(self) -> int:
        return max(self.tokens) + 1

    @classmethod
    def train(
        cls,
        corpus: Iterable[Sequence[UnitInput]],
        *,
        vocab_size: int | None = None,
        num_merges: int | None = None,
        min_count: int = 2,
        strict: bool = True,
        progress: bool = False,
    ) -> Self:
        if vocab_size is not None and vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if num_merges is not None and num_merges < 0:
            raise ValueError("num_merges must be non-negative")
        if min_count <= 0:
            raise ValueError("min_count must be positive")

        corpus_iter = _progress_iter(
            corpus,
            progress=progress,
            desc="CodecBPE corpus",
            unit="seq",
        )
        units = [tuple(_normalize_unit(unit) for unit in seq) for seq in corpus_iter]
        if not units:
            raise ValueError("corpus must not be empty")
        if strict and any(len(seq) == 0 for seq in units):
            raise ValueError("corpus must not contain empty sequences")
        units = [seq for seq in units if seq]
        if not units:
            raise ValueError("corpus must contain at least one unit")
        _unit_dim_from_sequences(units)

        base = sorted({unit for seq in units for unit in seq})
        unit_to_id = {unit: token_id for token_id, unit in enumerate(base)}
        tokens: dict[int, tuple[Unit, ...]] = {
            token_id: (unit,)
            for unit, token_id in unit_to_id.items()
        }
        encoded = [[unit_to_id[unit] for unit in seq] for seq in units]
        merges: list[Merge] = []
        next_id = len(base)
        if vocab_size is not None and next_id > vocab_size:
            raise ValueError("vocab_size must be at least the number of unique corpus units")

        progress_bar = _progress_bar(
            progress=progress,
            total=cls._merge_progress_total(next_id, vocab_size, num_merges),
            desc="CodecBPE merges",
            unit="merge",
        )
        stop_reason = "limit"
        try:
            while cls._can_merge(next_id, merges, vocab_size, num_merges):
                pair_counts = cls._count_pairs(encoded)
                if not pair_counts:
                    stop_reason = "no_pairs"
                    break

                best_count = max(pair_counts.values())
                if best_count < min_count:
                    stop_reason = "min_count"
                    break

                pair = min(pair for pair, count in pair_counts.items() if count == best_count)
                left, right = pair
                token_id = next_id
                next_id += 1

                tokens[token_id] = tokens[left] + tokens[right]
                merges.append(Merge(left=left, right=right, token_id=token_id))
                encoded = [cls._merge(seq, pair, token_id) for seq in encoded]
                if progress_bar is not None:
                    progress_bar.set_postfix(count=best_count, vocab=next_id)
                    progress_bar.update()
        finally:
            if progress_bar is not None:
                if stop_reason != "limit" and progress_bar.total is not None:
                    progress_bar.total = progress_bar.n
                progress_bar.set_postfix_str(f"stop={stop_reason}, vocab={next_id}")
                progress_bar.refresh()
                progress_bar.close()

        return cls(tokens, merges, strict=strict)

    def unit_ids(self, token_id: int, *, strict: bool | None = None) -> tuple[Unit, ...]:
        use_strict = self.strict if strict is None else strict
        if token_id in self.tokens:
            return self.tokens[token_id]
        if use_strict:
            raise KeyError(f"unknown token_id: {token_id}")
        if self.unit_dim is not None:
            raise KeyError(f"unknown token_id cannot be expanded for tuple units: {token_id}")
        return (token_id,)

    def encode_units(self, units: Sequence[UnitInput], *, strict: bool | None = None) -> list[int]:
        use_strict = self.strict if strict is None else strict
        if not units:
            if use_strict:
                raise ValueError("units must not be empty")
            return []

        token_ids: list[int] = []
        normalized = tuple(_normalize_unit(unit) for unit in units)
        _validate_unit_dim(normalized, self.unit_dim)
        for unit in normalized:
            token_id = self.unit_to_id.get(unit)
            if token_id is None:
                if use_strict:
                    raise KeyError(f"unknown unit: {unit}")
                if self.unit_dim is not None:
                    raise KeyError(f"unknown tuple unit cannot be encoded: {unit}")
                if unit in self.tokens:
                    raise ValueError(f"unknown unit collides with token_id: {unit}")
                token_ids.append(unit)
            else:
                token_ids.append(token_id)

        for merge in self.merges:
            token_ids = self._merge(token_ids, (merge.left, merge.right), merge.token_id)
        return token_ids

    def eval(self, corpus: Iterable[Sequence[UnitInput]]) -> CompressionStats:
        return _eval(corpus, self.encode_units)

    def expand_ids(self, token_ids: Sequence[int], *, strict: bool | None = None) -> list[Unit]:
        unit_ids, _ = self.expand_with_counts(token_ids, strict=strict)
        return unit_ids

    def expand_with_counts(
        self,
        token_ids: Sequence[int],
        *,
        strict: bool | None = None,
    ) -> tuple[list[Unit], list[int]]:
        use_strict = self.strict if strict is None else strict
        if not token_ids:
            if use_strict:
                raise ValueError("token_ids must not be empty")
            return [], []

        unit_ids: list[Unit] = []
        counts: list[int] = []
        for token_id in token_ids:
            units = self.unit_ids(token_id, strict=use_strict)
            unit_ids.extend(units)
            counts.append(len(units))
        return unit_ids, counts

    def repeat_interleave(
        self,
        x: torch.Tensor,
        token_ids: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        dim: int = -2,
        strict: bool | None = None,
    ) -> RepeatInterleaveOutput:
        if x.dim() == 0:
            raise ValueError("x must have at least one dimension")
        self._validate_token_ids(token_ids)

        dim = self._normalize_dim(dim, x.dim())
        if token_ids.dim() == 1:
            if mask is not None:
                raise ValueError("mask is only supported for 2D token_ids")
            return self._repeat_interleave_1d(x, token_ids, dim=dim, strict=strict)

        if token_ids.dim() == 2:
            return self._repeat_interleave_2d(
                x,
                token_ids,
                mask,
                dim=dim,
                strict=strict,
            )

        raise ValueError("token_ids must be a 1D or 2D tensor")

    def _repeat_interleave_1d(
        self,
        x: torch.Tensor,
        token_ids: torch.Tensor,
        *,
        dim: int,
        strict: bool | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x.size(dim) != token_ids.numel():
            raise ValueError("x and token_ids must align on the sequence dimension")

        unit_ids, counts = self.expand_with_counts(
            [int(token_id) for token_id in token_ids.tolist()],
            strict=strict,
        )
        repeats = torch.tensor(counts, dtype=torch.long, device=x.device)
        expanded_x = torch.repeat_interleave(x, repeats, dim=dim)
        expanded_ids = _unit_tensor(unit_ids, dtype=token_ids.dtype, device=token_ids.device)
        return expanded_x, expanded_ids

    def _repeat_interleave_2d(
        self,
        x: torch.Tensor,
        token_ids: torch.Tensor,
        mask: torch.Tensor | None,
        *,
        dim: int,
        strict: bool | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if dim == 0:
            raise ValueError("batched repeat_interleave expects a non-batch sequence dim")

        batch_size, seq_len = token_ids.shape
        if x.size(0) != batch_size:
            raise ValueError("x and token_ids must align on the batch dimension")
        if x.size(dim) != seq_len:
            raise ValueError("x and token_ids must align on the sequence dimension")

        token_mask = self._normalize_mask(mask, token_ids)
        pad_id = self._pad_id(token_ids, token_mask)

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
                strict=strict,
            )
            expanded_rows.append(expanded_x)
            expanded_id_rows.append(expanded_ids)

        row_lengths = [self._expanded_length(row) for row in expanded_id_rows]
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

        if pad_id is None:
            raise ValueError("cannot infer padding id from token_ids and mask")

        out_shape = list(x.shape)
        out_shape[dim] = max_len
        expanded_x = x.new_zeros(out_shape)
        ids_shape = (batch_size, max_len)
        if self.unit_dim is not None:
            ids_shape = (*ids_shape, self.unit_dim)
        expanded_ids = token_ids.new_full(ids_shape, pad_id)
        expanded_mask = torch.zeros(
            (batch_size, max_len),
            dtype=torch.bool,
            device=token_ids.device,
        )

        for row, (row_x, row_ids) in enumerate(zip(expanded_rows, expanded_id_rows, strict=True)):
            length = self._expanded_length(row_ids)
            row_target = expanded_x.select(0, row)
            slices = [slice(None)] * row_target.dim()
            slices[row_dim] = slice(0, length)
            row_target[tuple(slices)] = row_x
            expanded_ids[row, :length, ...] = row_ids
            expanded_mask[row, :length] = True

        return expanded_x, expanded_ids, expanded_mask

    def _expanded_length(self, unit_ids: torch.Tensor) -> int:
        if self.unit_dim is None:
            return unit_ids.numel()
        return unit_ids.size(0)

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
    def _pad_id(token_ids: torch.Tensor, mask: torch.Tensor) -> int | None:
        pad_values = token_ids.masked_select(~mask)
        if pad_values.numel() == 0:
            return None
        unique = torch.unique(pad_values)
        if unique.numel() != 1:
            raise ValueError("token_ids padding values must be identical")
        return int(unique.item())

    @staticmethod
    def _build_unit_to_id(tokens: Mapping[int, tuple[Unit, ...]]) -> dict[Unit, int]:
        unit_to_id: dict[Unit, int] = {}
        for token_id, units in tokens.items():
            if len(units) == 1:
                unit_to_id[units[0]] = token_id
        return unit_to_id

    @staticmethod
    def _can_merge(
        next_id: int,
        merges: Sequence[Merge],
        vocab_size: int | None,
        num_merges: int | None,
    ) -> bool:
        if vocab_size is not None and next_id >= vocab_size:
            return False
        return num_merges is None or len(merges) < num_merges

    @staticmethod
    def _merge_progress_total(
        next_id: int,
        vocab_size: int | None,
        num_merges: int | None,
    ) -> int | None:
        vocab_merges = None if vocab_size is None else max(vocab_size - next_id, 0)
        if num_merges is None:
            return vocab_merges
        if vocab_merges is None:
            return num_merges
        return min(vocab_merges, num_merges)

    @staticmethod
    def _count_pairs(seqs: Sequence[Sequence[int]]) -> Counter[tuple[int, int]]:
        counts: Counter[tuple[int, int]] = Counter()
        for seq in seqs:
            counts.update(zip(seq, seq[1:], strict=False))
        return counts

    @staticmethod
    def _merge(seq: Sequence[int], pair: tuple[int, int], token_id: int) -> list[int]:
        left, right = pair
        merged: list[int] = []
        index = 0
        while index < len(seq):
            if index + 1 < len(seq) and seq[index] == left and seq[index + 1] == right:
                merged.append(token_id)
                index += 2
            else:
                merged.append(seq[index])
                index += 1
        return merged

    @staticmethod
    def _normalize_dim(dim: int, ndim: int) -> int:
        if dim < 0:
            dim += ndim
        if dim < 0 or dim >= ndim:
            raise ValueError("dim is out of range for x")
        return dim


class CodecBPE:
    def __init__(
        self,
        *,
        cache_capacity: int | None = None,
        dropout: float | None = None,
        unk_token: str | None = None,
        continuing_subword_prefix: str | None = None,
        end_of_word_suffix: str | None = None,
        fuse_unk: bool | None = None,
        byte_fallback: bool = False,
        ignore_merges: bool = False,
    ) -> None:
        self.cache_capacity = cache_capacity
        self.dropout = dropout
        self.unk_token = unk_token
        self.continuing_subword_prefix = continuing_subword_prefix
        self.end_of_word_suffix = end_of_word_suffix
        self.fuse_unk = fuse_unk
        self.byte_fallback = byte_fallback
        self.ignore_merges = ignore_merges

        self._core: _CoreBPE | None = None
        self._unit_tokens: dict[Unit, str] | None = None
        self._token_texts: dict[int, str] | None = None
        self._model: TokenizersBPE | None = None

    @classmethod
    def from_dict(
        cls,
        state: CodecBPEState,
        *,
        cache_capacity: int | None = None,
        dropout: float | None = None,
        unk_token: str | None = None,
        continuing_subword_prefix: str | None = None,
        end_of_word_suffix: str | None = None,
        fuse_unk: bool | None = None,
        byte_fallback: bool = False,
        ignore_merges: bool = False,
    ) -> Self:
        bpe = cls(
            cache_capacity=cache_capacity,
            dropout=dropout,
            unk_token=unk_token,
            continuing_subword_prefix=continuing_subword_prefix,
            end_of_word_suffix=end_of_word_suffix,
            fuse_unk=fuse_unk,
            byte_fallback=byte_fallback,
            ignore_merges=ignore_merges,
        )
        tokens = {
            int(token_id): tuple(_unit_from_state(unit) for unit in units)
            for token_id, units in state["tokens"].items()
        }
        merges = [
            (
                int(merge["left"]),
                int(merge["right"]),
                int(merge["token_id"]),
            )
            for merge in state["merges"]
        ]
        strict = bool(state["strict"])
        bpe._bind_core(_CoreBPE(tokens, merges, strict=strict))
        return bpe

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        *,
        cache_capacity: int | None = None,
        dropout: float | None = None,
        unk_token: str | None = None,
        continuing_subword_prefix: str | None = None,
        end_of_word_suffix: str | None = None,
        fuse_unk: bool | None = None,
        byte_fallback: bool = False,
        ignore_merges: bool = False,
    ) -> Self:
        state_path = cls._state_path(path)
        state = json.loads(state_path.read_text(encoding="utf-8"))
        return cls.from_dict(
            state,
            cache_capacity=cache_capacity,
            dropout=dropout,
            unk_token=unk_token,
            continuing_subword_prefix=continuing_subword_prefix,
            end_of_word_suffix=end_of_word_suffix,
            fuse_unk=fuse_unk,
            byte_fallback=byte_fallback,
            ignore_merges=ignore_merges,
        )

    @classmethod
    def train(
        cls,
        corpus: Iterable[Sequence[UnitInput]],
        *,
        vocab_size: int | None = None,
        num_merges: int | None = None,
        min_count: int = 2,
        strict: bool = True,
        progress: bool = False,
        cache_capacity: int | None = None,
        dropout: float | None = None,
        unk_token: str | None = None,
        continuing_subword_prefix: str | None = None,
        end_of_word_suffix: str | None = None,
        fuse_unk: bool | None = None,
        byte_fallback: bool = False,
        ignore_merges: bool = False,
    ) -> Self:
        core = _CoreBPE.train(
            corpus,
            vocab_size=vocab_size,
            num_merges=num_merges,
            min_count=min_count,
            strict=strict,
            progress=progress,
        )
        bpe = cls(
            cache_capacity=cache_capacity,
            dropout=dropout,
            unk_token=unk_token,
            continuing_subword_prefix=continuing_subword_prefix,
            end_of_word_suffix=end_of_word_suffix,
            fuse_unk=fuse_unk,
            byte_fallback=byte_fallback,
            ignore_merges=ignore_merges,
        )
        bpe._bind_core(core)
        return bpe

    @property
    def core(self) -> _CoreBPE:
        return self._require_core()

    @property
    def model(self) -> TokenizersBPE:
        return self._require_model()

    @property
    def tokens(self) -> Mapping[int, tuple[Unit, ...]]:
        return self.core.tokens

    @property
    def merges(self) -> tuple[Merge, ...]:
        return self.core.merges

    @property
    def vocab_size(self) -> int:
        return self.core.vocab_size

    @property
    def strict(self) -> bool:
        return self.core.strict

    def tokenizer(self) -> Tokenizer:
        tokenizer_type = _require_tokenizer()
        return tokenizer_type(self.model)

    def units_text(self, units: Sequence[UnitInput]) -> str:
        normalized = tuple(_normalize_unit(unit) for unit in units)
        _validate_unit_dim(normalized, self.core.unit_dim)
        return self._units_text(normalized, self._require_unit_tokens())

    def token_text(self, token_id: int) -> str:
        token_texts = self._require_token_texts()
        if token_id not in token_texts:
            raise KeyError(f"unknown token_id: {token_id}")
        return token_texts[token_id]

    def encode_units(self, units: Sequence[UnitInput]) -> list[int]:
        core = self.core
        if not units:
            if core.strict:
                raise ValueError("units must not be empty")
            return []

        normalized = tuple(_normalize_unit(unit) for unit in units)
        _validate_unit_dim(normalized, core.unit_dim)
        unit_tokens = self._require_unit_tokens()
        if not core.strict and any(unit not in unit_tokens for unit in normalized):
            return core.encode_units(units, strict=False)

        text = self._units_text(normalized, unit_tokens)
        return [token.id for token in self.model.tokenize(text)]

    def eval(self, corpus: Iterable[Sequence[UnitInput]]) -> CompressionStats:
        return _eval(corpus, self.encode_units)

    def expand_ids(self, token_ids: Sequence[int], *, strict: bool | None = None) -> list[Unit]:
        return self.core.expand_ids(token_ids, strict=strict)

    def expand_with_counts(
        self,
        token_ids: Sequence[int],
        *,
        strict: bool | None = None,
    ) -> tuple[list[Unit], list[int]]:
        return self.core.expand_with_counts(token_ids, strict=strict)

    def repeat_interleave(
        self,
        x: torch.Tensor,
        token_ids: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        dim: int = -2,
        strict: bool | None = None,
    ) -> RepeatInterleaveOutput:
        return self.core.repeat_interleave(x, token_ids, mask, dim=dim, strict=strict)

    def to_dict(self) -> CodecBPEState:
        return {
            "tokens": {
                str(token_id): [_unit_to_state(unit) for unit in units]
                for token_id, units in sorted(self.tokens.items())
            },
            "merges": [
                {
                    "left": merge.left,
                    "right": merge.right,
                    "token_id": merge.token_id,
                }
                for merge in self.merges
            ],
            "strict": self.strict,
        }

    def save_pretrained(self, path: str | Path) -> Path:
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        (out / "codec_bpe.json").write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        self.tokenizer().save(str(out / "tokenizer.json"))
        return out

    def _bind_core(self, core: _CoreBPE) -> None:
        tokenizers_bpe = _require_tokenizers_bpe()
        unit_tokens = self._build_unit_tokens(core)
        token_texts = {
            token_id: self._units_text(units, unit_tokens)
            for token_id, units in core.tokens.items()
        }
        vocab = {text: token_id for token_id, text in token_texts.items()}
        merges = [
            (token_texts[merge.left], token_texts[merge.right])
            for merge in core.merges
        ]

        self._core = core
        self._unit_tokens = unit_tokens
        self._token_texts = token_texts
        self._model = tokenizers_bpe(**self._tokenizers_kwargs(vocab, merges))

    @staticmethod
    def _build_unit_tokens(core: _CoreBPE) -> dict[Unit, str]:
        units = sorted(core.unit_to_id)
        if len(units) > _private_use_capacity():
            raise ValueError("too many units for CodecBPE")
        return {
            unit: _private_use_char(index)
            for index, unit in enumerate(units)
        }

    def _require_core(self) -> _CoreBPE:
        if self._core is None:
            raise ValueError("CodecBPE is not initialized; use train(), from_dict(), or from_pretrained()")
        return self._core

    def _require_model(self) -> TokenizersBPE:
        if self._model is None:
            raise ValueError("CodecBPE is not initialized; use train(), from_dict(), or from_pretrained()")
        return self._model

    def _require_unit_tokens(self) -> Mapping[Unit, str]:
        if self._unit_tokens is None:
            raise ValueError("CodecBPE is not initialized; use train(), from_dict(), or from_pretrained()")
        return self._unit_tokens

    def _require_token_texts(self) -> Mapping[int, str]:
        if self._token_texts is None:
            raise ValueError("CodecBPE is not initialized; use train(), from_dict(), or from_pretrained()")
        return self._token_texts

    def _tokenizers_kwargs(
        self,
        vocab: dict[str, int],
        merges: list[tuple[str, str]],
    ) -> _TokenizersBPEKwargs:
        kwargs: _TokenizersBPEKwargs = {
            "vocab": vocab,
            "merges": merges,
            "byte_fallback": self.byte_fallback,
            "ignore_merges": self.ignore_merges,
        }
        if self.cache_capacity is not None:
            kwargs["cache_capacity"] = self.cache_capacity
        if self.dropout is not None:
            kwargs["dropout"] = self.dropout
        if self.unk_token is not None:
            kwargs["unk_token"] = self.unk_token
        if self.continuing_subword_prefix is not None:
            kwargs["continuing_subword_prefix"] = self.continuing_subword_prefix
        if self.end_of_word_suffix is not None:
            kwargs["end_of_word_suffix"] = self.end_of_word_suffix
        if self.fuse_unk is not None:
            kwargs["fuse_unk"] = self.fuse_unk
        return kwargs

    @staticmethod
    def _state_path(path: str | Path) -> Path:
        state_path = Path(path)
        if state_path.is_dir():
            return state_path / "codec_bpe.json"
        return state_path

    @staticmethod
    def _units_text(units: Sequence[Unit], unit_tokens: Mapping[Unit, str]) -> str:
        pieces: list[str] = []
        for unit in units:
            if unit not in unit_tokens:
                raise KeyError(f"unknown unit: {unit}")
            pieces.append(unit_tokens[unit])
        return "".join(pieces)


def _require_tokenizers_bpe() -> type[TokenizersBPE]:
    try:
        from tokenizers.models import BPE as TokenizersBPE
    except ImportError as error:
        raise ImportError("CodecBPE requires the `tokenizers` package") from error
    return TokenizersBPE


def _require_tokenizer() -> type[Tokenizer]:
    try:
        from tokenizers import Tokenizer
    except ImportError as error:
        raise ImportError("CodecBPE.tokenizer() requires the `tokenizers` package") from error
    return Tokenizer


def _progress_iter[T](
    iterable: Iterable[T],
    *,
    progress: bool,
    desc: str,
    unit: str,
) -> Iterable[T]:
    if not progress:
        return iterable
    tqdm = _require_tqdm()
    total = len(iterable) if isinstance(iterable, Sized) else None
    return tqdm(iterable, total=total, desc=desc, unit=unit)


def _progress_bar(
    *,
    progress: bool,
    total: int | None,
    desc: str,
    unit: str,
) -> Tqdm | None:
    if not progress:
        return None
    tqdm = _require_tqdm()
    return tqdm(total=total, desc=desc, unit=unit)


def _require_tqdm() -> type[Tqdm]:
    try:
        from tqdm.auto import tqdm
    except ImportError as error:
        raise ImportError("CodecBPE progress requires the `tqdm` package") from error
    return tqdm


def _normalize_unit(unit: UnitInput) -> Unit:
    if isinstance(unit, int):
        if unit < 0:
            raise ValueError("unit ids must be non-negative")
        return unit
    if isinstance(unit, str):
        raise TypeError("tuple units must contain integer ids")

    values = tuple(int(value) for value in unit)
    if not values:
        raise ValueError("tuple units must not be empty")
    if any(value < 0 for value in values):
        raise ValueError("unit ids must be non-negative")
    return values


def _unit_dim_from_sequences(seqs: Sequence[Sequence[Unit]]) -> int | None:
    return _unit_dim(unit for seq in seqs for unit in seq)


def _unit_dim_from_token_units(token_units: Iterable[Sequence[Unit]]) -> int | None:
    return _unit_dim(unit for units in token_units for unit in units)


def _unit_dim(units: Iterable[Unit]) -> int | None:
    dim: int | None = None
    initialized = False
    for unit in units:
        current = len(unit) if isinstance(unit, tuple) else None
        if not initialized:
            dim = current
            initialized = True
            continue
        if dim != current:
            raise ValueError("units must be all ints or fixed-length integer sequences")
    return dim


def _validate_unit_dim(units: Sequence[Unit], dim: int | None) -> None:
    inferred = _unit_dim(units)
    if inferred != dim:
        raise ValueError("units must match the tokenizer unit shape")


def _unit_to_state(unit: Unit) -> UnitState:
    if isinstance(unit, tuple):
        return list(unit)
    return unit


def _unit_from_state(unit: UnitState) -> Unit:
    return _normalize_unit(unit)


def _unit_tensor(
    units: Sequence[Unit],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    return torch.tensor(units, dtype=dtype, device=device)


def _private_use_capacity() -> int:
    return sum(end - start + 1 for start, end in PRIVATE_USE_RANGES)


def _private_use_char(index: int) -> str:
    for start, end in PRIVATE_USE_RANGES:
        size = end - start + 1
        if index < size:
            return chr(start + index)
        index -= size
    raise ValueError("private-use character index out of range")


def _eval(
    corpus: Iterable[Sequence[UnitInput]],
    encode_units: Callable[[Sequence[UnitInput]], Sequence[int]],
) -> CompressionStats:
    num_sequences = 0
    original_tokens = 0
    encoded_tokens = 0
    for seq in corpus:
        units = tuple(_normalize_unit(unit) for unit in seq)
        num_sequences += 1
        original_tokens += len(units)
        encoded_tokens += len(encode_units(units))

    if num_sequences == 0:
        raise ValueError("corpus must not be empty")
    if original_tokens == 0:
        raise ValueError("corpus must contain at least one unit")

    compression_ratio = encoded_tokens / original_tokens
    compression_factor = original_tokens / encoded_tokens if encoded_tokens else float("inf")
    return CompressionStats(
        num_sequences=num_sequences,
        original_tokens=original_tokens,
        encoded_tokens=encoded_tokens,
        mean_original_length=original_tokens / num_sequences,
        mean_encoded_length=encoded_tokens / num_sequences,
        compression_ratio=compression_ratio,
        compression_factor=compression_factor,
        compression_gain=1.0 - compression_ratio,
    )
