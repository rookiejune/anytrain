from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

import torch
import torch.nn.functional as F
from lightning import pytorch as pl
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset

from anytrain.hydra import run_train


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


def main():
    from omegaconf import OmegaConf

    config_path = Path(__file__).with_name("configs") / "tiny_regression.yaml"
    run_train(OmegaConf.load(config_path))


if __name__ == "__main__":
    main()
