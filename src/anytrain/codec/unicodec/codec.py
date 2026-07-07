from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any, Union

import torch
from torch import Tensor, nn
from typing_extensions import TypeAlias

from .assets import UniCodecAssets, ensure_unicodec_assets

Domain: TypeAlias = Union[str, int]


class UniCodec(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        assets: UniCodecAssets,
    ) -> None:
        super().__init__()

        self.model = model
        self.device = device
        self.assets = assets

    @classmethod
    def from_pretrained(
        cls,
        *,
        cache_dir: str | os.PathLike[str] | None = None,
        repo_id: str | None = None,
        config_name: str | None = None,
        checkpoint_filename: str | None = None,
        device: str | torch.device | None = None,
        local_files_only: bool = False,
        force_download: bool = False,
    ) -> UniCodec:
        kwargs: dict[str, object] = {
            "cache_dir": cache_dir,
            "local_files_only": local_files_only,
            "force_download": force_download,
        }
        if repo_id is not None:
            kwargs["repo_id"] = repo_id
        if config_name is not None:
            kwargs["config_name"] = config_name
        if checkpoint_filename is not None:
            kwargs["checkpoint_filename"] = checkpoint_filename

        assets = ensure_unicodec_assets(**kwargs)
        model_cls = _load_unicodec_model()
        resolved_device = _resolve_device(device)
        model = model_cls.from_pretrained0802(
            str(assets["config"]),
            str(assets["checkpoint"]),
        )
        model = model.to(resolved_device)
        model.eval()

        return cls(model=model, device=resolved_device, assets=assets)

    @torch.no_grad()
    def encode(
        self,
        audio: Tensor,
        domain: Domain | Sequence[Domain],
        *,
        bandwidth_id: int | Tensor = 0,
    ) -> Tensor:
        _, codes = self.encode_features(audio, domain, bandwidth_id=bandwidth_id)
        return codes

    @torch.no_grad()
    def encode_features(
        self,
        audio: Tensor,
        domain: Domain | Sequence[Domain],
        *,
        bandwidth_id: int | Tensor = 0,
    ) -> tuple[Tensor, Tensor]:
        if audio.dim() != 2:
            raise ValueError("UniCodec encode expects audio shape [batch, time].")

        domains = _resolve_domains(domain, audio.size(0))
        cond = _resolve_bandwidth_id(bandwidth_id, self.device)
        features, codes = self.model.encode_infer(
            audio.to(self.device),
            domains,
            bandwidth_id=cond,
        )
        return features, codes

    @torch.no_grad()
    def decode(
        self,
        codes: Tensor,
        *,
        bandwidth_id: int | Tensor = 0,
    ) -> Tensor:
        return self.decode_features(
            self.codes_to_features(codes),
            bandwidth_id=bandwidth_id,
        )

    @torch.no_grad()
    def decode_features(
        self,
        features: Tensor,
        *,
        bandwidth_id: int | Tensor = 0,
    ) -> Tensor:
        cond = _resolve_bandwidth_id(bandwidth_id, self.device)
        return self.model.decode(features.to(self.device), bandwidth_id=cond)

    @torch.no_grad()
    def codes_to_features(self, codes: Tensor) -> Tensor:
        return self.model.codes_to_features(codes.to(self.device))

    @torch.no_grad()
    def reconstruct(
        self,
        audio: Tensor,
        domain: Domain | Sequence[Domain],
        *,
        bandwidth_id: int | Tensor = 0,
    ) -> Tensor:
        codes = self.encode(audio, domain, bandwidth_id=bandwidth_id)
        return self.decode(codes, bandwidth_id=bandwidth_id)


def _resolve_domains(domain: Domain | Sequence[Domain], batch_size: int) -> tuple[str, ...]:
    if isinstance(domain, str) or isinstance(domain, int):
        domains = (domain,) * batch_size
    else:
        domains = tuple(domain)
        if len(domains) != batch_size:
            raise ValueError("UniCodec domain sequence must match audio batch size.")

    resolved = tuple(_resolve_domain(item) for item in domains)
    return resolved


def _resolve_domain(domain: Domain) -> str:
    value = str(domain)
    if value not in {"0", "1", "2"}:
        raise ValueError("UniCodec domain must be one of '0', '1', or '2'.")
    return value


def _resolve_bandwidth_id(bandwidth_id: int | Tensor, device: torch.device) -> Tensor:
    if isinstance(bandwidth_id, Tensor):
        return bandwidth_id.to(device)
    return torch.tensor([bandwidth_id], device=device)


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_unicodec_model() -> Any:
    try:
        from unicodec import Unicodec
    except ImportError as exc:
        raise ImportError(
            "UniCodec requires the installable UniCodec fork. Install "
            "`anytrain[unicodec]` or install "
            "`unicodec @ git+https://github.com/rookiejune/UniCodec.git`."
        ) from exc
    return Unicodec
