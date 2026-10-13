"""``OptionScorer``: scores every option independently and normalises across options.

Each option is encoded as its own ``(prompt, option)`` sequence by a shared backbone and
mapped to a scalar by a small head; a softmax over the (unpadded) options gives the
decision distribution. Options never attend to each other, so the output is equivariant to
option order by construction (checked in ``tests/test_model.py`` and at eval time).

A natural upgrade is the packed layout used by the reference implementation: one sequence
with a shared prefix and a block attention mask, which saves recomputing the prompt.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import torch
from torch import nn

from .backbones import HFBackbone, ToyBackbone, pool
from .tokenizer import PairTokenizer, build_tokenizer


@dataclass
class ModelConfig:
    tokenizer: str = "byte"  # "byte" | "hf:<name_or_path>"
    backbone: str = "toy"  # "toy" | "hf:<name_or_path>"
    max_length: int = 128
    pooling: str = "mean"  # mean | cls | last
    head_dropout: float = 0.1
    # toy backbone only
    hidden_size: int = 64
    num_layers: int = 2
    num_heads: int = 4
    # hf backbone only
    freeze_backbone: bool = False
    lora: dict | None = field(default=None)

    @classmethod
    def from_dict(cls, d: dict) -> ModelConfig:
        return cls(**d)

    def to_dict(self) -> dict:
        return asdict(self)


class OptionScorer(nn.Module):
    def __init__(self, cfg: ModelConfig, tokenizer: PairTokenizer):
        super().__init__()
        self.cfg = cfg
        if cfg.backbone == "toy":
            self.backbone = ToyBackbone(
                tokenizer.vocab_size, cfg.hidden_size, cfg.num_layers, cfg.num_heads, cfg.max_length
            )
        elif cfg.backbone.startswith("hf:"):
            self.backbone = HFBackbone(cfg.backbone[3:], cfg.freeze_backbone, cfg.lora)
        else:
            raise ValueError(f"unknown backbone {cfg.backbone!r}; use 'toy' or 'hf:<name_or_path>'")
        h = self.backbone.hidden_size
        self.head = nn.Sequential(
            nn.Dropout(cfg.head_dropout), nn.Linear(h, h), nn.GELU(), nn.Linear(h, 1)
        )

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, option_mask: torch.Tensor
    ) -> torch.Tensor:
        """Returns logits ``(B, K)``; padded options get ``-inf``."""
        b, k, length = input_ids.shape
        flat_valid = option_mask.reshape(-1)
        ids = input_ids.reshape(b * k, length)[flat_valid]
        mask = attention_mask.reshape(b * k, length)[flat_valid]  # only real options are encoded
        hidden = self.backbone(ids, mask)
        scores = self.head(pool(hidden, mask, self.cfg.pooling)).squeeze(-1)
        logits = torch.full((b * k,), float("-inf"), dtype=scores.dtype, device=scores.device)
        logits[flat_valid] = scores
        return logits.reshape(b, k)


def build_model(cfg: ModelConfig | dict) -> tuple[OptionScorer, PairTokenizer]:
    cfg = ModelConfig.from_dict(cfg) if isinstance(cfg, dict) else cfg
    tokenizer = build_tokenizer(cfg.tokenizer)
    return OptionScorer(cfg, tokenizer), tokenizer
