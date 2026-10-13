"""Train with PyTorch Lightning.

    python -m rlcd.train --config configs/smoke.yaml
    python -m rlcd.train --config configs/default.yaml --set trainer.max_epochs=3 data.train_path=my.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from .config import load_config
from .data import RLCDDataModule
from .lit import RLCDModule
from .model import ModelConfig


def train(cfg: dict) -> str:
    """Runs fit (and test if ``data.test_path`` is set). Returns the best checkpoint path."""
    L.seed_everything(cfg.get("seed", 0), workers=True)
    model_cfg = ModelConfig.from_dict(cfg.get("model", {}))
    dm = RLCDDataModule(
        tokenizer=model_cfg.tokenizer,
        max_length=model_cfg.max_length,
        seed=cfg.get("seed", 0),
        **cfg.get("data", {}),
    )
    module = RLCDModule(model_cfg=model_cfg.to_dict(), loss_cfg=cfg.get("loss"), **cfg.get("optim", {}))

    trainer_cfg = dict(cfg.get("trainer", {}))
    root = Path(trainer_cfg.pop("default_root_dir", "outputs"))
    ckpt = ModelCheckpoint(monitor="val/loss", mode="min", save_top_k=1, save_last=True,
                           filename="best-{epoch}-{step}")
    trainer = L.Trainer(
        default_root_dir=root,
        logger=CSVLogger(root, name="logs") if cfg.get("log", True) else False,
        callbacks=[ckpt, LearningRateMonitor(logging_interval="step")],
        **trainer_cfg,
    )
    trainer.fit(module, datamodule=dm)
    if cfg.get("data", {}).get("test_path"):
        trainer.test(module, datamodule=dm, ckpt_path="best")
    return ckpt.best_model_path


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides")
    args = p.parse_args(argv)
    best = train(load_config(args.config, args.set))
    print(f"best checkpoint: {best}")


if __name__ == "__main__":
    main()
