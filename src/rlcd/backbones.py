"""Sequence encoders. Each maps ``(N, L)`` ids/mask to hidden states ``(N, L, H)``."""

from __future__ import annotations

import torch
from torch import nn


class ToyBackbone(nn.Module):
    """Small from-scratch transformer encoder. No downloads; for tests and smoke runs."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        max_length: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.tok = nn.Embedding(vocab_size, hidden_size)
        self.pos = nn.Embedding(max_length, hidden_size)
        layer = nn.TransformerEncoderLayer(
            hidden_size, num_heads, 4 * hidden_size, dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(input_ids.shape[1], device=input_ids.device)
        x = self.tok(input_ids) + self.pos(pos)[None]
        x = self.encoder(x, src_key_padding_mask=attention_mask == 0)
        return self.norm(x)


class HFBackbone(nn.Module):
    """Any Hugging Face encoder (``AutoModel``), optionally frozen and/or wrapped with LoRA."""

    def __init__(self, name_or_path: str, freeze: bool = False, lora: dict | None = None):
        super().__init__()
        from transformers import AutoModel

        # transformers>=5 loads in the checkpoint's dtype (often bf16); keep fp32 so the head matches
        # outside autocast and mixed precision is left to the Lightning ``precision`` setting.
        self.model = AutoModel.from_pretrained(name_or_path).float()
        self.hidden_size = int(self.model.config.hidden_size)
        if freeze or lora:
            for p in self.model.parameters():
                p.requires_grad = False
        if lora:
            from peft import LoraConfig, get_peft_model

            cfg = {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "target_modules": "all-linear"}
            cfg.update(lora)
            self.model = get_peft_model(self.model, LoraConfig(**cfg))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state


def pool(hidden: torch.Tensor, attention_mask: torch.Tensor, how: str) -> torch.Tensor:
    """Reduce ``(N, L, H)`` to ``(N, H)``. Assumes right padding."""
    if how == "mean":
        m = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * m).sum(1) / m.sum(1).clamp_min(1)
    if how == "cls":
        return hidden[:, 0]
    if how == "last":
        last = attention_mask.sum(-1).sub(1).clamp_min(0)
        return hidden[torch.arange(hidden.shape[0], device=hidden.device), last]
    raise ValueError(f"unknown pooling {how!r}; use mean, cls or last")
