from __future__ import annotations

import torch
from lightning import pytorch as pl


class StopOnNonfiniteLossCallback(pl.Callback):
    def on_before_backward(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        loss: torch.Tensor,
    ) -> None:
        loss_to_check = loss.detach()
        if torch.isfinite(loss_to_check).all():
            return

        raise RuntimeError(
            "Non-finite loss detected before backward "
            f"(epoch={trainer.current_epoch}, global_step={trainer.global_step}, "
            f"loss={loss_to_check})."
        )
