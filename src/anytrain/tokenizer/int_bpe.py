from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, NotRequired, Self, TypedDict

import torch

if TYPE_CHECKING:
    from tokenizers import Tokenizer
    from tokenizers.models import BPE as TokenizersBPE


PRIVATE_USE_START = 0xE000
PRIVATE_USE_END = 0xF8FF


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


class IntBPEState(TypedDict):
    tokens: dict[str, list[int]]
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


class _CoreBPE:
    def __init__(
        self,
        tokens: Mapping[int, Sequence[int]],
        merges: Sequence[Merge | tuple[int, int, int]] = (),
        *,
        strict: bool = True,
    ) -> None:
        if not tokens:
            raise ValueError("tokens must not be empty")

        self.strict = strict
        self.tokens = {
            int(token_id): tuple(int(unit) for unit in units)
            for token_id, units in tokens.items()
        }
        for token_id, units in self.tokens.items():
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
        corpus: Iterable[Sequence[int]],
        *,
        vocab_size: int | None = None,
        num_merges: int | None = None,
        min_count: int = 2,
        strict: bool = True,
    ) -> Self:
        if vocab_size is not None and vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if num_merges is not None and num_merges < 0:
            raise ValueError("num_merges must be non-negative")
        if min_count <= 0:
            raise ValueError("min_count must be positive")

        units = [tuple(int(unit) for unit in seq) for seq in corpus]
        if not units:
            raise ValueError("corpus must not be empty")
        if strict and any(len(seq) == 0 for seq in units):
            raise ValueError("corpus must not contain empty sequences")
        units = [seq for seq in units if seq]
        if not units:
            raise ValueError("corpus must contain at least one unit")

        base = sorted({unit for seq in units for unit in seq})
        tokens: dict[int, tuple[int, ...]] = {unit: (unit,) for unit in base}
        encoded = [list(seq) for seq in units]
        merges: list[Merge] = []
        next_id = max(base) + 1

        while cls._can_merge(tokens, merges, vocab_size, num_merges):
            pair_counts = cls._count_pairs(encoded)
            if not pair_counts:
                break

            best_count = max(pair_counts.values())
            if best_count < min_count:
                break

            pair = min(pair for pair, count in pair_counts.items() if count == best_count)
            left, right = pair
            token_id = next_id
            next_id += 1

            tokens[token_id] = tokens[left] + tokens[right]
            merges.append(Merge(left=left, right=right, token_id=token_id))
            encoded = [cls._merge(seq, pair, token_id) for seq in encoded]

        return cls(tokens, merges, strict=strict)

    def unit_ids(self, token_id: int, *, strict: bool | None = None) -> tuple[int, ...]:
        use_strict = self.strict if strict is None else strict
        if token_id in self.tokens:
            return self.tokens[token_id]
        if use_strict:
            raise KeyError(f"unknown token_id: {token_id}")
        return (token_id,)

    def encode_units(self, units: Sequence[int], *, strict: bool | None = None) -> list[int]:
        use_strict = self.strict if strict is None else strict
        if not units:
            if use_strict:
                raise ValueError("units must not be empty")
            return []

        token_ids: list[int] = []
        for unit in units:
            token_id = self.unit_to_id.get(unit)
            if token_id is None:
                if use_strict:
                    raise KeyError(f"unknown unit: {unit}")
                token_ids.append(unit)
            else:
                token_ids.append(token_id)

        for merge in self.merges:
            token_ids = self._merge(token_ids, (merge.left, merge.right), merge.token_id)
        return token_ids

    def eval(self, corpus: Iterable[Sequence[int]]) -> CompressionStats:
        return _eval(corpus, self.encode_units)

    def expand_ids(self, token_ids: Sequence[int], *, strict: bool | None = None) -> list[int]:
        unit_ids, _ = self.expand_with_counts(token_ids, strict=strict)
        return unit_ids

    def expand_with_counts(
        self,
        token_ids: Sequence[int],
        *,
        strict: bool | None = None,
    ) -> tuple[list[int], list[int]]:
        use_strict = self.strict if strict is None else strict
        if not token_ids:
            if use_strict:
                raise ValueError("token_ids must not be empty")
            return [], []

        unit_ids: list[int] = []
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
        *,
        dim: int = -2,
        strict: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if token_ids.dim() != 1:
            raise ValueError("token_ids must be a 1D tensor")
        if x.dim() == 0:
            raise ValueError("x must have at least one dimension")
        if token_ids.dtype == torch.bool or torch.is_floating_point(token_ids) or torch.is_complex(token_ids):
            raise TypeError("token_ids must contain integer ids")

        dim = self._normalize_dim(dim, x.dim())
        if x.size(dim) != token_ids.numel():
            raise ValueError("x and token_ids must align on the sequence dimension")

        unit_ids, counts = self.expand_with_counts(
            [int(token_id) for token_id in token_ids.tolist()],
            strict=strict,
        )
        repeats = torch.tensor(counts, dtype=torch.long, device=x.device)
        expanded_x = torch.repeat_interleave(x, repeats, dim=dim)
        expanded_ids = torch.tensor(unit_ids, dtype=token_ids.dtype, device=token_ids.device)
        return expanded_x, expanded_ids

    @staticmethod
    def _build_unit_to_id(tokens: Mapping[int, tuple[int, ...]]) -> dict[int, int]:
        unit_to_id: dict[int, int] = {}
        for token_id, units in tokens.items():
            if len(units) == 1:
                unit_to_id[units[0]] = token_id
        return unit_to_id

    @staticmethod
    def _can_merge(
        tokens: Mapping[int, tuple[int, ...]],
        merges: Sequence[Merge],
        vocab_size: int | None,
        num_merges: int | None,
    ) -> bool:
        if vocab_size is not None and len(tokens) >= vocab_size:
            return False
        return num_merges is None or len(merges) < num_merges

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


class IntBPE:
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
        self._unit_tokens: dict[int, str] | None = None
        self._token_texts: dict[int, str] | None = None
        self._model: TokenizersBPE | None = None

    @classmethod
    def from_dict(
        cls,
        state: IntBPEState,
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
            int(token_id): tuple(int(unit) for unit in units)
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
        corpus: Iterable[Sequence[int]],
        *,
        vocab_size: int | None = None,
        num_merges: int | None = None,
        min_count: int = 2,
        strict: bool = True,
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
    def tokens(self) -> Mapping[int, tuple[int, ...]]:
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

    def units_text(self, units: Sequence[int]) -> str:
        return self._units_text(tuple(int(unit) for unit in units), self._require_unit_tokens())

    def token_text(self, token_id: int) -> str:
        token_texts = self._require_token_texts()
        if token_id not in token_texts:
            raise KeyError(f"unknown token_id: {token_id}")
        return token_texts[token_id]

    def encode_units(self, units: Sequence[int]) -> list[int]:
        core = self.core
        if not units:
            if core.strict:
                raise ValueError("units must not be empty")
            return []

        unit_tokens = self._require_unit_tokens()
        if not core.strict and any(unit not in unit_tokens for unit in units):
            return core.encode_units(units, strict=False)

        text = self.units_text(units)
        return [token.id for token in self.model.tokenize(text)]

    def eval(self, corpus: Iterable[Sequence[int]]) -> CompressionStats:
        return _eval(corpus, self.encode_units)

    def expand_ids(self, token_ids: Sequence[int], *, strict: bool | None = None) -> list[int]:
        return self.core.expand_ids(token_ids, strict=strict)

    def expand_with_counts(
        self,
        token_ids: Sequence[int],
        *,
        strict: bool | None = None,
    ) -> tuple[list[int], list[int]]:
        return self.core.expand_with_counts(token_ids, strict=strict)

    def repeat_interleave(
        self,
        x: torch.Tensor,
        token_ids: torch.Tensor,
        *,
        dim: int = -2,
        strict: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.core.repeat_interleave(x, token_ids, dim=dim, strict=strict)

    def to_dict(self) -> IntBPEState:
        return {
            "tokens": {
                str(token_id): list(units)
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
        (out / "int_bpe.json").write_text(
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
    def _build_unit_tokens(core: _CoreBPE) -> dict[int, str]:
        units = sorted(core.unit_to_id)
        if len(units) > PRIVATE_USE_END - PRIVATE_USE_START + 1:
            raise ValueError("too many units for IntBPE")
        return {
            unit: chr(PRIVATE_USE_START + index)
            for index, unit in enumerate(units)
        }

    def _require_core(self) -> _CoreBPE:
        if self._core is None:
            raise ValueError("IntBPE is not initialized; use train(), from_dict(), or from_pretrained()")
        return self._core

    def _require_model(self) -> TokenizersBPE:
        if self._model is None:
            raise ValueError("IntBPE is not initialized; use train(), from_dict(), or from_pretrained()")
        return self._model

    def _require_unit_tokens(self) -> Mapping[int, str]:
        if self._unit_tokens is None:
            raise ValueError("IntBPE is not initialized; use train(), from_dict(), or from_pretrained()")
        return self._unit_tokens

    def _require_token_texts(self) -> Mapping[int, str]:
        if self._token_texts is None:
            raise ValueError("IntBPE is not initialized; use train(), from_dict(), or from_pretrained()")
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
            return state_path / "int_bpe.json"
        return state_path

    @staticmethod
    def _units_text(units: Sequence[int], unit_tokens: Mapping[int, str]) -> str:
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
        raise ImportError("IntBPE requires the `tokenizers` package") from error
    return TokenizersBPE


def _require_tokenizer() -> type[Tokenizer]:
    try:
        from tokenizers import Tokenizer
    except ImportError as error:
        raise ImportError("IntBPE.tokenizer() requires the `tokenizers` package") from error
    return Tokenizer


def _eval(
    corpus: Iterable[Sequence[int]],
    encode_units: Callable[[Sequence[int]], Sequence[int]],
) -> CompressionStats:
    num_sequences = 0
    original_tokens = 0
    encoded_tokens = 0
    for seq in corpus:
        units = tuple(int(unit) for unit in seq)
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
