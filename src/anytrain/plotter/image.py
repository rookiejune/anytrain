from __future__ import annotations

from typing import Any, Literal

import torch

INSTALL_HINT = "Install plot dependencies with `pip install anytrain[plot]`."
ChannelOrder = Literal["auto", "chw", "hwc"]


class TensorImagePlotter:
    def __init__(
        self,
        *,
        cmap: str = "viridis",
        title: str | None = None,
        channel_order: ChannelOrder = "auto",
    ) -> None:
        if not cmap:
            raise ValueError("cmap must not be empty.")
        if title == "":
            raise ValueError("title must not be empty.")
        if channel_order not in ("auto", "chw", "hwc"):
            raise ValueError("channel_order must be 'auto', 'chw', or 'hwc'.")
        self.cmap = cmap
        self.title = title
        self.channel_order = channel_order

    def __call__(self, image: torch.Tensor) -> Any:
        data = _image_array(image, channel_order=self.channel_order)
        pyplot = _load_pyplot()
        figure, axes = pyplot.subplots()
        if data.ndim == 2:
            axes.imshow(data, cmap=self.cmap)
        else:
            axes.imshow(data)
        if self.title is not None:
            axes.set_title(self.title)
        axes.axis("off")
        figure.tight_layout()
        return figure


def _image_array(image: torch.Tensor, *, channel_order: ChannelOrder) -> Any:
    if not isinstance(image, torch.Tensor):
        raise TypeError("image must be a torch.Tensor.")
    if image.ndim not in (2, 3):
        raise ValueError("image must be a single 2D grayscale or 3D channel image tensor.")
    if image.numel() == 0:
        raise ValueError("image must not be empty.")

    data = image.detach().cpu()
    if not torch.is_floating_point(data):
        data = data.float()

    if data.ndim == 2:
        return data.numpy()

    data = _to_hwc(data, channel_order=channel_order)
    if data.shape[-1] == 1:
        data = data[..., 0]
    return data.numpy()


def _to_hwc(image: torch.Tensor, *, channel_order: ChannelOrder) -> torch.Tensor:
    if channel_order == "chw":
        return _chw_to_hwc(image)
    if channel_order == "hwc":
        return _validate_hwc(image)
    if image.shape[0] in (1, 3, 4):
        return _chw_to_hwc(image)
    if image.shape[-1] in (1, 3, 4):
        return image
    raise ValueError("3D image tensor must be CHW or HWC with 1, 3, or 4 channels.")


def _chw_to_hwc(image: torch.Tensor) -> torch.Tensor:
    if image.shape[0] not in (1, 3, 4):
        raise ValueError("CHW image tensor must have 1, 3, or 4 channels.")
    return image.movedim(0, -1)


def _validate_hwc(image: torch.Tensor) -> torch.Tensor:
    if image.shape[-1] not in (1, 3, 4):
        raise ValueError("HWC image tensor must have 1, 3, or 4 channels.")
    return image


def _load_pyplot() -> Any:
    try:
        import matplotlib.pyplot as pyplot
    except ImportError as exc:
        raise ImportError(
            f"`anytrain.plotter.image` requires `matplotlib`. {INSTALL_HINT}"
        ) from exc
    return pyplot
