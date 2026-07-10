from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import TypedDict

import torch

from anytrain._compat import Self

from ._core import CoreBPE, base_text, private_use_tokens
from ._eval import EvalStats, evaluate
from ._eval import top_k as eval_top_k
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
        from tokenizers.models import BPE

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
    ) -> Self:
        codec = FrameCodec(codebook_sizes)
        core = CoreBPE.train(
            lambda: ([codec.encode(frame) for frame in frames] for frames in corpus()),
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

        codec = self._codec
        base_ids = tuple(codec.encode(frame) for frame in frames)
        base_tokens = self._base_tokens
        for base_id in base_ids:
            if base_id not in base_tokens:
                raise KeyError(f"unknown frame: {codec.decode(base_id)}")

        text = base_text(base_ids, base_tokens)
        return [token.id for token in self._model.tokenize(text)]

    def decode(self, token_ids: Sequence[int]) -> list[Frame]:
        base_ids, _ = self._core.decode_with_counts(token_ids)
        return [self._codec.decode(base_id) for base_id in base_ids]

    def repeat_interleave(
        self,
        x: torch.Tensor,
        token_ids: torch.Tensor,
        *,
        dim: int = -2,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expanded_x, base_ids = self._core.repeat_interleave(x, token_ids, dim=dim)
        if self._codec.num_codebooks == 1:
            return expanded_x, base_ids.unsqueeze(-1)

        flat_ids = [int(value) for value in base_ids.detach().cpu().tolist()]
        frames = [self._codec.decode(base_id) for base_id in flat_ids]
        frame_tensor = torch.tensor(frames, dtype=base_ids.dtype, device=base_ids.device)
        return expanded_x, frame_tensor

    def eval(
        self,
        corpus: Iterable[Sequence[Sequence[int]]],
        *,
        show_progress: bool = True,
        top_k: int = 100,
    ) -> EvalStats:
        return evaluate(
            corpus,
            self._codec,
            self.encode,
            token_lengths=self._core.token_lengths(),
            vocab_size=self.vocab_size,
            show_progress=show_progress,
            top_k=eval_top_k(top_k),
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
