"""RLCD-style objective: maximise a proper scoring reward.

    reward = w_log * log(p_y) + w_sph * p_y / ||p||_2 - w_rps * RPS(p, y)   (RPS on ordinal rows only)

A positive combination of proper scoring rules is itself proper, so the expected reward is
maximised by reporting the true probabilities, which is what makes the outputs calibrated.
The loss is the negative mean reward.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ScoringWeights:
    log: float = 1.0
    spherical: float = 0.5
    rps: float = 1.0


def scoring_reward(
    logits: torch.Tensor,
    label: torch.Tensor,
    option_mask: torch.Tensor,
    ordinal: torch.Tensor,
    weights: ScoringWeights | None = None,
) -> dict[str, torch.Tensor]:
    """Per-example reward and its components. ``logits`` is (B, K) with -inf on padded options."""
    w = weights or ScoringWeights()
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = log_probs.exp()  # exactly 0 on padded options
    idx = label.unsqueeze(-1)
    log_score = log_probs.gather(-1, idx).squeeze(-1)
    spherical = probs.gather(-1, idx).squeeze(-1) / probs.pow(2).sum(-1).sqrt()

    # Ranked probability score over ordered options, normalised by (K - 1).
    k = option_mask.sum(-1)
    pos = torch.arange(logits.shape[-1], device=logits.device)
    cdf_p = probs.cumsum(-1)
    cdf_y = (pos[None] >= label[:, None]).to(probs.dtype)
    in_range = pos[None] < (k[:, None] - 1)
    rps = ((cdf_p - cdf_y).pow(2) * in_range).sum(-1) / (k - 1).clamp_min(1)
    rps = rps * ordinal.to(rps.dtype)

    reward = w.log * log_score + w.spherical * spherical - w.rps * rps
    return {"reward": reward, "log_score": log_score, "spherical": spherical, "rps": rps, "probs": probs}


def rlcd_loss(
    logits: torch.Tensor,
    label: torch.Tensor,
    option_mask: torch.Tensor,
    ordinal: torch.Tensor,
    weights: ScoringWeights | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    out = scoring_reward(logits, label, option_mask, ordinal, weights)
    return -out["reward"].mean(), out
