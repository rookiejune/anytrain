from __future__ import annotations

import os
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from .assets import LongCatAssets, LongCatDecoderName, ensure_longcat_assets

DEFAULT_DECODER: LongCatDecoderName = "16k_4codebooks"


class LongCatAudioCodec(nn.Module):
    def __init__(
        self,
        encoder: Any,
        decoders: dict[str, nn.Module],
        device: torch.device,
        assets: LongCatAssets,
    ) -> None:
        super().__init__()

        self.encoder = encoder
        self.decoders = nn.ModuleDict(decoders)
        self.device = device
        self.assets = assets

    @property
    def semantic_codebook(self) -> Tensor:
        codebook = self.encoder.get_semantic_codebook()
        if not isinstance(codebook, Tensor):
            raise TypeError("LongCat semantic codebook must be a Tensor.")
        if codebook.dim() != 2:
            raise ValueError("LongCat semantic codebook must have shape [dim, vocab].")
        return codebook.transpose(0, 1).contiguous().to(self.device)

    @classmethod
    def from_pretrained(
        cls,
        *,
        cache_dir: str | os.PathLike[str] | None = None,
        decoders: Sequence[LongCatDecoderName] = (DEFAULT_DECODER,),
        device: str | torch.device | None = None,
        local_files_only: bool = False,
        force_download: bool = False,
    ) -> LongCatAudioCodec:
        load_encoder, load_decoder = _load_longcat_loaders()
        decoder_names = _validate_decoders(decoders)
        assets = ensure_longcat_assets(
            cache_dir=cache_dir,
            decoders=decoder_names,
            local_files_only=local_files_only,
            force_download=force_download,
        )
        resolved_device = _resolve_device(device)

        with _longcat_checkpoint_env(assets.ckpt_dir):
            encoder = load_encoder(str(assets.configs.encoder), resolved_device)
            loaded_decoders: dict[str, nn.Module] = {
                name: load_decoder(str(assets.configs.decoder(name)), resolved_device)
                for name in decoder_names
            }

        return cls(
            encoder=encoder,
            decoders=loaded_decoders,
            device=resolved_device,
            assets=assets,
        )

    @torch.no_grad()
    def encode(
        self,
        audio: Tensor,
        sample_rate: int,
        *,
        n_acoustic_codebooks: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        codes = self.encoder(
            audio.to(self.device),
            sample_rate,
            n_acoustic_codebooks=n_acoustic_codebooks,
        )
        return codes[0], codes[1]

    @torch.no_grad()
    def decode(
        self,
        semantic_codes: Tensor,
        acoustic_codes: Tensor,
        *,
        decoder: LongCatDecoderName = DEFAULT_DECODER,
    ) -> Tensor:
        model = self._decoder(decoder)
        return model(
            semantic_codes.to(self.device),
            acoustic_codes.to(self.device),
        )

    @torch.no_grad()
    def acoustic_codes_to_features(
        self,
        acoustic_codes: Tensor,
        *,
        decoder: LongCatDecoderName = DEFAULT_DECODER,
    ) -> Tensor:
        model = self._decoder(decoder)
        convert = getattr(model, "acoustic_codes_to_latents", None)
        if not callable(convert):
            raise TypeError("LongCat decoder must provide acoustic_codes_to_latents().")
        features = convert(acoustic_codes.to(self.device))
        if not isinstance(features, Tensor):
            raise TypeError("LongCat acoustic_codes_to_latents() must return a Tensor.")
        if features.dim() != 3:
            raise ValueError("LongCat acoustic features must have shape [batch, dim, time].")
        if not torch.is_floating_point(features) or torch.is_complex(features):
            raise TypeError("LongCat acoustic features must be floating point tensors.")
        return features.transpose(1, 2)

    @torch.no_grad()
    def decode_features(
        self,
        semantic_codes: Tensor,
        acoustic_features: Tensor,
        *,
        decoder: LongCatDecoderName = DEFAULT_DECODER,
    ) -> Tensor:
        if semantic_codes.dim() != 2:
            raise ValueError("semantic_codes must have shape [batch, time].")
        if (
            semantic_codes.dtype == torch.bool
            or torch.is_floating_point(semantic_codes)
            or torch.is_complex(semantic_codes)
        ):
            raise TypeError("semantic_codes must contain integer ids.")
        if acoustic_features.dim() != 3:
            raise ValueError("acoustic_features must have shape [batch, time, dim].")
        if acoustic_features.shape[:2] != semantic_codes.shape:
            raise ValueError("semantic_codes and acoustic_features must align on batch and time.")
        if not torch.is_floating_point(acoustic_features) or torch.is_complex(acoustic_features):
            raise TypeError("acoustic_features must be floating point tensors.")

        model = self._decoder(decoder)
        return model(
            semantic_codes.to(self.device),
            acoustic_features.to(self.device).transpose(1, 2),
        )

    @torch.no_grad()
    def reconstruct(
        self,
        audio: Tensor,
        sample_rate: int,
        *,
        decoder: LongCatDecoderName = DEFAULT_DECODER,
        n_acoustic_codebooks: int | None = None,
    ) -> Tensor:
        semantic_codes, acoustic_codes = self.encode(
            audio,
            sample_rate,
            n_acoustic_codebooks=n_acoustic_codebooks,
        )
        return self.decode(semantic_codes, acoustic_codes, decoder=decoder)

    def _decoder(self, name: LongCatDecoderName) -> Any:
        try:
            return self.decoders[name]
        except KeyError as exc:
            available = ", ".join(sorted(self.decoders))
            raise ValueError(
                f"Decoder {name!r} is not loaded. Available decoders: {available}."
            ) from exc


def _validate_decoders(decoders: Sequence[LongCatDecoderName]) -> tuple[LongCatDecoderName, ...]:
    if not decoders:
        raise ValueError("decoders must not be empty.")

    valid: set[LongCatDecoderName] = {
        "16k_4codebooks",
        "24k_2codebooks",
        "24k_4codebooks",
    }
    unknown = [name for name in decoders if name not in valid]
    if unknown:
        raise ValueError(f"Unknown LongCat decoders: {unknown}.")

    return tuple(dict.fromkeys(decoders))


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_longcat_loaders():
    try:
        from longcat_audio_codec import load_decoder, load_encoder
    except ImportError as exc:
        raise ImportError(
            "LongCatAudioCodec requires the installable LongCat fork. Install "
            "`anytrain[longcat]` or install "
            "`longcat-audio-codec @ git+https://github.com/rookiejune/LongCat-Audio-Codec.git`."
        ) from exc
    return load_encoder, load_decoder


@contextmanager
def _longcat_checkpoint_env(ckpt_dir: Path):
    name = "LONGCAT_AUDIO_CODEC_CKPT_DIR"
    old = os.environ.get(name)
    os.environ[name] = str(ckpt_dir)
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old
