from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, overload

import torch

from anytrain._compat import Self

from .core import _CoreBPE
from .corpus import (
    _corpus_factory,
    _encoded_corpus,
    _private_use_capacity,
    _private_use_char,
)
from .eval import _eval, _normalize_top_k
from .frame import _FrameCodec
from .interop import _require_tokenizer, _require_tokenizers_bpe
from .stats import CodecBPEEvalStats, Merge
from .types import (
    CodecBPEState,
    Frame,
    FrameCorpus,
    FrameCorpusFactory,
    FrameInput,
    RepeatInterleaveOutput,
    _TokenizersBPEKwargs,
)

if TYPE_CHECKING:
    from tokenizers import Tokenizer
    from tokenizers.models import BPE as TokenizersBPE


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

        self._codec: _FrameCodec | None = None
        self._core: _CoreBPE | None = None
        self._base_tokens: dict[int, str] | None = None
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
        codec = _FrameCodec(state["codebook_sizes"])
        tokens = {
            int(token_id): tuple(codec.encode(frame) for frame in frames)
            for token_id, frames in state["tokens"].items()
        }
        merges = [
            (
                int(merge["left"]),
                int(merge["right"]),
                int(merge["token_id"]),
            )
            for merge in state["merges"]
        ]
        bpe._bind_core(_CoreBPE(tokens, merges), codec)
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
        corpus: FrameCorpus | FrameCorpusFactory,
        *,
        codebook_sizes: Sequence[int],
        vocab_size: int = 30_000,
        min_frequency: int = 0,
        show_progress: bool = True,
        max_token_length: int | None = None,
        cache_capacity: int | None = None,
        dropout: float | None = None,
        unk_token: str | None = None,
        continuing_subword_prefix: str | None = None,
        end_of_word_suffix: str | None = None,
        fuse_unk: bool | None = None,
        byte_fallback: bool = False,
        ignore_merges: bool = False,
    ) -> Self:
        codec = _FrameCodec(codebook_sizes)
        corpus_factory = _corpus_factory(corpus)
        core = _CoreBPE.train(
            lambda: _encoded_corpus(corpus_factory(), codec),
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            show_progress=show_progress,
            max_token_length=max_token_length,
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
        bpe._bind_core(core, codec)
        return bpe

    @property
    def model(self) -> TokenizersBPE:
        return self._require_model()

    @property
    def codebook_sizes(self) -> tuple[int, ...]:
        return self._require_codec().codebook_sizes

    @property
    def num_codebooks(self) -> int:
        return self._require_codec().num_codebooks

    @property
    def tokens(self) -> Mapping[int, tuple[Frame, ...]]:
        codec = self._require_codec()
        core = self._require_core()
        return {
            token_id: tuple(codec.decode(base_id) for base_id in base_ids)
            for token_id, base_ids in core.tokens.items()
        }

    @property
    def merges(self) -> tuple[Merge, ...]:
        return self._require_core().merges

    @property
    def vocab_size(self) -> int:
        return self._require_core().vocab_size

    def tokenizer(self) -> Tokenizer:
        tokenizer_type = _require_tokenizer()
        return tokenizer_type(self.model)

    def frames_text(self, frames: Sequence[FrameInput]) -> str:
        codec = self._require_codec()
        base_ids = tuple(codec.encode(frame) for frame in frames)
        return self._base_text(base_ids, self._require_base_tokens())

    def token_text(self, token_id: int) -> str:
        token_texts = self._require_token_texts()
        if token_id not in token_texts:
            raise KeyError(f"unknown token_id: {token_id}")
        return token_texts[token_id]

    @overload
    def encode(self, frames: Sequence[FrameInput]) -> list[int]: ...

    @overload
    def encode(self, frames: torch.Tensor) -> torch.Tensor: ...

    def encode(self, frames: Sequence[FrameInput] | torch.Tensor) -> list[int] | torch.Tensor:
        if isinstance(frames, torch.Tensor):
            sequence = self._tensor_to_frames(frames)
            token_ids = self.encode(sequence)
            return torch.tensor(token_ids, dtype=torch.long, device=frames.device)
        return self._encode_sequence(frames)

    def _encode_sequence(self, frames: Sequence[FrameInput]) -> list[int]:
        if not frames:
            raise ValueError("frames must not be empty")

        codec = self._require_codec()
        base_ids = tuple(codec.encode(frame) for frame in frames)
        base_tokens = self._require_base_tokens()
        if any(base_id not in base_tokens for base_id in base_ids):
            missing = next(base_id for base_id in base_ids if base_id not in base_tokens)
            raise KeyError(f"unknown frame: {codec.decode(missing)}")

        text = self._base_text(base_ids, base_tokens)
        return [token.id for token in self.model.tokenize(text)]

    def eval(
        self,
        corpus: Iterable[Sequence[FrameInput]],
        *,
        show_progress: bool = True,
        top_k: int = 100,
    ) -> CodecBPEEvalStats:
        return _eval(
            corpus,
            self._require_codec(),
            self._encode_sequence,
            token_lengths=self._require_core()._token_lengths(),
            vocab_size=self.vocab_size,
            show_progress=show_progress,
            top_k=_normalize_top_k(top_k),
        )

    @overload
    def decode(self, token_ids: Sequence[int]) -> list[Frame]: ...

    @overload
    def decode(self, token_ids: torch.Tensor) -> torch.Tensor: ...

    def decode(
        self,
        token_ids: Sequence[int] | torch.Tensor,
    ) -> list[Frame] | torch.Tensor:
        frames, _ = self.decode_with_counts(token_ids)
        return frames

    @overload
    def decode_with_counts(
        self,
        token_ids: Sequence[int],
    ) -> tuple[list[Frame], list[int]]: ...

    @overload
    def decode_with_counts(
        self,
        token_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def decode_with_counts(
        self,
        token_ids: Sequence[int] | torch.Tensor,
    ) -> tuple[list[Frame], list[int]] | tuple[torch.Tensor, torch.Tensor]:
        if isinstance(token_ids, torch.Tensor):
            return self._decode_tensor(token_ids)
        return self._decode_sequence(token_ids)

    def _decode_sequence(self, token_ids: Sequence[int]) -> tuple[list[Frame], list[int]]:
        codec = self._require_codec()
        base_ids, counts = self._require_core().decode_with_counts(token_ids)
        return [codec.decode(base_id) for base_id in base_ids], counts

    def _decode_tensor(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._require_core()._validate_token_ids(token_ids)
        if token_ids.dim() != 1:
            raise ValueError("token ids tensor must have shape [tokens]")

        ids = [int(token_id) for token_id in token_ids.detach().cpu().tolist()]
        frames, counts = self._decode_sequence(ids)
        return (
            self._frames_to_tensor(frames, device=token_ids.device),
            torch.tensor(counts, dtype=torch.long, device=token_ids.device),
        )

    def repeat_interleave(
        self,
        x: torch.Tensor,
        token_ids: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        dim: int = -2,
    ) -> RepeatInterleaveOutput:
        expanded = self._require_core().repeat_interleave(x, token_ids, mask, dim=dim)
        if len(expanded) == 2:
            expanded_x, base_ids = expanded
            return expanded_x, self._base_ids_to_frames_tensor(base_ids)
        expanded_x, base_ids, expanded_mask = expanded
        return expanded_x, self._base_ids_to_frames_tensor(base_ids), expanded_mask

    def to_dict(self) -> CodecBPEState:
        codec = self._require_codec()
        core = self._require_core()
        return {
            "codebook_sizes": list(codec.codebook_sizes),
            "tokens": {
                str(token_id): [list(codec.decode(base_id)) for base_id in base_ids]
                for token_id, base_ids in sorted(core.tokens.items())
            },
            "merges": [
                {
                    "left": merge.left,
                    "right": merge.right,
                    "token_id": merge.token_id,
                }
                for merge in self.merges
            ],
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

    def _bind_core(self, core: _CoreBPE, codec: _FrameCodec) -> None:
        tokenizers_bpe = _require_tokenizers_bpe()
        base_tokens = self._build_base_tokens(core)
        token_texts = {
            token_id: self._base_text(base_ids, base_tokens)
            for token_id, base_ids in core.tokens.items()
        }
        vocab = {text: token_id for token_id, text in token_texts.items()}
        merges = [(token_texts[merge.left], token_texts[merge.right]) for merge in core.merges]

        self._codec = codec
        self._core = core
        self._base_tokens = base_tokens
        self._token_texts = token_texts
        self._model = tokenizers_bpe(**self._tokenizers_kwargs(vocab, merges))

    @staticmethod
    def _build_base_tokens(core: _CoreBPE) -> dict[int, str]:
        base_ids = sorted(core.base_to_id)
        if len(base_ids) > _private_use_capacity():
            raise ValueError("too many observed frames for CodecBPE")
        return {base_id: _private_use_char(index) for index, base_id in enumerate(base_ids)}

    def _tensor_to_frames(self, frames: torch.Tensor) -> list[list[int]]:
        self._require_core()._validate_token_ids(frames)
        codec = self._require_codec()
        if codec.num_codebooks == 1:
            if frames.dim() != 1:
                raise ValueError("single-codebook tensor input must have shape [time]")
            return [[int(value)] for value in frames.detach().cpu().tolist()]

        if frames.dim() != 2 or frames.size(1) != codec.num_codebooks:
            raise ValueError("multi-codebook tensor input must have shape [time, codebooks]")
        return [[int(value) for value in frame] for frame in frames.detach().cpu().tolist()]

    def _frames_to_tensor(
        self,
        frames: Sequence[Frame],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        codec = self._require_codec()
        if codec.num_codebooks == 1:
            return torch.tensor(
                [frame[0] for frame in frames],
                dtype=torch.long,
                device=device,
            )

        if not frames:
            return torch.empty((0, codec.num_codebooks), dtype=torch.long, device=device)
        return torch.tensor(frames, dtype=torch.long, device=device)

    def _base_ids_to_frames_tensor(self, base_ids: torch.Tensor) -> torch.Tensor:
        codec = self._require_codec()
        if codec.num_codebooks == 1:
            return base_ids.unsqueeze(-1)

        frames: list[Frame] = []
        flat_ids = [int(value) for value in base_ids.detach().cpu().reshape(-1).tolist()]
        for base_id in flat_ids:
            frames.append(codec.decode(base_id))
        return torch.tensor(
            frames,
            dtype=base_ids.dtype,
            device=base_ids.device,
        ).reshape(*base_ids.shape, codec.num_codebooks)

    def _require_codec(self) -> _FrameCodec:
        if self._codec is None:
            raise ValueError(
                "CodecBPE is not initialized; use train(), from_dict(), or from_pretrained()"
            )
        return self._codec

    def _require_core(self) -> _CoreBPE:
        if self._core is None:
            raise ValueError(
                "CodecBPE is not initialized; use train(), from_dict(), or from_pretrained()"
            )
        return self._core

    def _require_model(self) -> TokenizersBPE:
        if self._model is None:
            raise ValueError(
                "CodecBPE is not initialized; use train(), from_dict(), or from_pretrained()"
            )
        return self._model

    def _require_base_tokens(self) -> Mapping[int, str]:
        if self._base_tokens is None:
            raise ValueError(
                "CodecBPE is not initialized; use train(), from_dict(), or from_pretrained()"
            )
        return self._base_tokens

    def _require_token_texts(self) -> Mapping[int, str]:
        if self._token_texts is None:
            raise ValueError(
                "CodecBPE is not initialized; use train(), from_dict(), or from_pretrained()"
            )
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
    def _base_text(base_ids: Sequence[int], base_tokens: Mapping[int, str]) -> str:
        pieces: list[str] = []
        for base_id in base_ids:
            if base_id not in base_tokens:
                raise KeyError(f"unknown frame id: {base_id}")
            pieces.append(base_tokens[base_id])
        return "".join(pieces)
