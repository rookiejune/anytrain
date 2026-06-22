from __future__ import annotations

from collections.abc import Callable, Iterable
from functools import partial

import torch
import torch.nn.functional as F
from lightning import pytorch as pl
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset


class TinyRegressionModule(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        lr: float = 0.01,
        optimizer: Callable[[Iterable[torch.nn.Parameter]], Optimizer] | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.optimizer = optimizer

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = F.mse_loss(pred, y)
        self.log("train/loss", loss)
        return loss

    def configure_optimizers(self):
        if self.optimizer is not None:
            return self.optimizer(self.parameters())
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class TinyRegressionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        num_samples: int = 64,
        input_dim: int = 4,
        batch_size: int = 16,
    ) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.batch_size = batch_size

    def setup(self, stage: str | None = None) -> None:
        generator = torch.Generator().manual_seed(0)
        x = torch.randn(self.num_samples, self.input_dim, generator=generator)
        y = x.sum(dim=-1, keepdim=True)
        self.dataset = TensorDataset(x, y)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size)


def run_tiny_regression(
    *,
    max_epochs: int = 1,
    num_samples: int = 64,
    input_dim: int = 4,
    batch_size: int = 16,
    default_root_dir: str = "outputs/anytrain/tiny_regression",
    enable_progress_bar: bool = True,
):
    torch.set_float32_matmul_precision("medium")
    pl.seed_everything(0, workers=True)

    module = TinyRegressionModule(
        model=torch.nn.Linear(input_dim, 1),
        optimizer=partial(torch.optim.Adam, lr=0.01),
    )
    data_module = TinyRegressionDataModule(
        num_samples=num_samples,
        input_dim=input_dim,
        batch_size=batch_size,
    )
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=enable_progress_bar,
        default_root_dir=default_root_dir,
    )
    trainer.fit(module, datamodule=data_module)
    return trainer, module


def main():
    run_tiny_regression()


if __name__ == "__main__":
    main()
