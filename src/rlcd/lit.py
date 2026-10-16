"""Lightning training module."""

from __future__ import annotations

import math

import lightning as L
import torch

from .losses import ScoringWeights, rlcd_loss
from .metrics import MetricAccumulator
from .model import ModelConfig, build_model


class RLCDModule(L.LightningModule):
    def __init__(
        self,
        model_cfg: dict | None = None,
        loss_cfg: dict | None = None,
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 100,
        min_lr_ratio: float = 0.1,
    ):
        super().__init__()
        model_cfg = ModelConfig.from_dict(model_cfg or {}).to_dict()  # fill defaults before saving
        self.save_hyperparameters({
            "model_cfg": model_cfg, "loss_cfg": loss_cfg or {}, "lr": lr,
            "weight_decay": weight_decay, "warmup_steps": warmup_steps, "min_lr_ratio": min_lr_ratio,
        })
        self.model, _ = build_model(model_cfg)
        self.weights = ScoringWeights(**(loss_cfg or {}))
        self._val_metrics = MetricAccumulator()

    def forward(self, batch: dict) -> torch.Tensor:
        return self.model(batch["input_ids"], batch["attention_mask"], batch["option_mask"])

    def _step(self, batch: dict):
        logits = self(batch)
        loss, out = rlcd_loss(logits, batch["label"], batch["option_mask"], batch["ordinal"], self.weights)
        acc = (out["probs"].argmax(-1) == batch["label"]).float().mean()
        return loss, acc, out

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss, acc, _ = self._step(batch)
        n = batch["label"].shape[0]
        self.log_dict({"train/loss": loss, "train/acc": acc}, prog_bar=True, batch_size=n)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        loss, acc, out = self._step(batch)
        n = batch["label"].shape[0]
        self.log_dict({"val/loss": loss, "val/acc": acc}, prog_bar=True, batch_size=n, sync_dist=True)
        self._val_metrics.update(
            out["probs"], batch["label"], batch["option_mask"], batch["ordinal"], batch["source"]
        )

    def test_step(self, batch: dict, batch_idx: int) -> None:
        loss, acc, _ = self._step(batch)
        self.log_dict({"test/loss": loss, "test/acc": acc}, batch_size=batch["label"].shape[0])

    def on_validation_epoch_end(self) -> None:
        # Epoch-level calibration metrics. Computed per rank (not all-gathered).
        overall = self._val_metrics.compute()["overall"]
        self._val_metrics = MetricAccumulator()
        if overall["n"]:
            self.log_dict({f"val/{k}": overall[k] for k in ("nll", "brier", "ece")})

    def configure_optimizers(self):
        decay, no_decay = [], []
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                (no_decay if p.ndim < 2 or "norm" in name.lower() else decay).append(p)
        opt = torch.optim.AdamW(
            [{"params": decay, "weight_decay": self.hparams.weight_decay},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=self.hparams.lr,
        )
        total = max(1, int(self.trainer.estimated_stepping_batches))
        warmup, floor = self.hparams.warmup_steps, self.hparams.min_lr_ratio

        def schedule(step: int) -> float:  # linear warmup, then cosine to ``floor * lr``
            if step < warmup:
                return (step + 1) / max(1, warmup)
            t = (step - warmup) / max(1, total - warmup)
            return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, schedule)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}
