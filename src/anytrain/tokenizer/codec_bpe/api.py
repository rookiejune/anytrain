from __future__ import annotations

import json
import operator
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import TypedDict

from anytrain._compat import Self

from ._core import CoreBPE, base_text, private_use_capacity, private_use_tokens
from ._deps import bpe_class
from ._eval import EvalStats
from ._eval import evaluate as run_eval
from ._frame import Frame, FrameCodec


class CodecBPEState(TypedDict):
    codebook_sizes: list[int]
    tokens: dict[str, list[list[int]]]
    merges: list[dict[str, int]]


class CodecBPE:
    def __init__(self, core: CoreBPE, codec: FrameCodec) -> None:
        self._codec = codec
        self._core = core
        self._base_tokens = private_use_tokens(core.base_to_id)

        token_texts = {
            token_id: base_text(base_ids, self._base_tokens)
            for token_id, base_ids in core.tokens.items()
        }
        vocab = {text: token_id for token_id, text in token_texts.items()}
        merges = [(token_texts[left], token_texts[right]) for left, right, _ in core.merges]
        BPE = bpe_class()
        self._model = BPE(vocab=vocab, merges=merges)

    @classmethod
    def from_pretrained(cls, path: str | Path) -> Self:
        state_path = Path(path) / "codec_bpe.json"
        state: CodecBPEState = json.loads(state_path.read_text(encoding="utf-8"))
        codec = FrameCodec(state["codebook_sizes"])
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
        return cls(CoreBPE(tokens, merges), codec)

    @classmethod
    def train(
        cls,
        corpus: Callable[[], Iterable[Sequence[Sequence[int]]]],
        *,
        codebook_sizes: Sequence[int],
        vocab_size: int = 30_000,
        min_frequency: int = 0,
        show_progress: bool = True,
        max_token_length: int | None = None,
        max_frames: int | None = 1_000_000_000,
    ) -> Self:
        if max_frames is not None:
            if isinstance(max_frames, bool):
                raise TypeError("max_frames must be an integer or None")
            try:
                max_frames = operator.index(max_frames)
            except TypeError as error:
                raise TypeError("max_frames must be an integer or None") from error
            if max_frames <= 0:
                raise ValueError("max_frames must be positive or None")

        codec = FrameCodec(codebook_sizes)
        core = CoreBPE.train(
            lambda: _encode_corpus(
                corpus(),
                codec=codec,
                max_frames=max_frames,
            ),
            base=(
                range(codec.vocab_size)
                if codec.num_codebooks == 1
                and codec.vocab_size <= private_use_capacity()
                else None
            ),
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            show_progress=show_progress,
            max_token_length=max_token_length,
        )
        return cls(core, codec)

    @property
    def codebook_sizes(self) -> tuple[int, ...]:
        return self._codec.codebook_sizes

    @property
    def vocab_size(self) -> int:
        return self._core.vocab_size

    def encode(self, frames: Sequence[Sequence[int]]) -> list[int]:
        if not frames:
            raise ValueError("frames must not be empty")

        base_ids = tuple(self._codec.encode(frame) for frame in frames)
        text = base_text(base_ids, self._base_tokens)
        return [token.id for token in self._model.tokenize(text)]

    def decode(self, token_ids: Sequence[int]) -> list[Frame]:
        base_ids = self._core.decode(token_ids)
        return [self._codec.decode(base_id) for base_id in base_ids]

    def evaluate(
        self,
        corpus: Iterable[Sequence[Sequence[int]]],
        *,
        show_progress: bool = True,
        top_k: int = 100,
    ) -> EvalStats:
        return run_eval(
            corpus,
            self.encode,
            token_lengths=self._core.token_lengths(),
            vocab_size=self.vocab_size,
            show_progress=show_progress,
            top_k=top_k,
        )

    def save_pretrained(self, path: str | Path) -> Path:
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        state: CodecBPEState = {
            "codebook_sizes": list(self._codec.codebook_sizes),
            "tokens": {
                str(token_id): [list(self._codec.decode(base_id)) for base_id in base_ids]
                for token_id, base_ids in sorted(self._core.tokens.items())
            },
            "merges": [
                {
                    "left": left,
                    "right": right,
                    "token_id": token_id,
                }
                for left, right, token_id in self._core.merges
            ],
        }
        (out / "codec_bpe.json").write_text(
            json.dumps(state, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return out


def _encode_corpus(
    corpus: Iterable[Sequence[Sequence[int]]],
    *,
    codec: FrameCodec,
    max_frames: int | None,
) -> Iterable[list[int]]:
    """Encode complete sequences until their frame count reaches the limit."""

    remaining = max_frames
    for frames in corpus:
        base_ids = [codec.encode(frame) for frame in frames]
        yield base_ids
        if remaining is not None:
            remaining -= len(base_ids)
            if remaining <= 0:
                return
