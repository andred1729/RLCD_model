"""Decision-quality and calibration metrics over probability tensors (plain PyTorch)."""

from __future__ import annotations

import torch


def expected_calibration_error(conf: torch.Tensor, correct: torch.Tensor, bins: int = 15) -> float:
    edges = torch.linspace(0, 1, bins + 1)
    ece = torch.zeros(())
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi)
        if sel.any():
            ece += sel.float().mean() * (conf[sel].mean() - correct[sel].float().mean()).abs()
    return float(ece)


def compute_metrics(
    probs: torch.Tensor, label: torch.Tensor, option_mask: torch.Tensor, ordinal: torch.Tensor
) -> dict[str, float]:
    """``probs`` (N, K) is zero on padded options."""
    n = probs.shape[0]
    if n == 0:
        return {"n": 0}
    conf, pred = probs.max(-1)
    correct = pred == label
    p_y = probs.gather(-1, label[:, None]).squeeze(-1)
    onehot = torch.nn.functional.one_hot(label, probs.shape[-1]).to(probs.dtype)
    out = {
        "n": n,
        "accuracy": correct.float().mean().item(),
        "nll": -p_y.clamp_min(1e-12).log().mean().item(),
        "brier": (probs - onehot).pow(2).sum(-1).mean().item(),
        "mean_confidence": conf.mean().item(),
        "ece": expected_calibration_error(conf, correct),
    }
    if ordinal.any():
        k = option_mask.sum(-1)
        pos = torch.arange(probs.shape[-1])
        cdf_y = (pos[None] >= label[:, None]).to(probs.dtype)
        in_range = pos[None] < (k[:, None] - 1)
        rps = ((probs.cumsum(-1) - cdf_y).pow(2) * in_range).sum(-1) / (k - 1).clamp_min(1)
        out["rps_ordinal"] = rps[ordinal].mean().item()
    return out


class MetricAccumulator:
    """Collects predictions across batches; ``compute`` reports overall and per-source metrics."""

    def __init__(self) -> None:
        self._probs: list[torch.Tensor] = []
        self._label: list[torch.Tensor] = []
        self._mask: list[torch.Tensor] = []
        self._ordinal: list[torch.Tensor] = []
        self._source: list[str] = []

    def update(self, probs, label, option_mask, ordinal, source: list[str] | None = None) -> None:
        self._probs.append(probs.detach().cpu().float())
        self._mask.append(option_mask.detach().cpu())
        self._label.append(label.detach().cpu())
        self._ordinal.append(ordinal.detach().cpu())
        self._source += source or ["default"] * probs.shape[0]

    def compute(self) -> dict:
        if not self._probs:
            return {"overall": {"n": 0}, "by_source": {}}
        k = max(p.shape[-1] for p in self._probs)
        def pad(t):  # batches may have different option counts
            return torch.nn.functional.pad(t, (0, k - t.shape[-1]))

        probs = torch.cat([pad(p) for p in self._probs])
        mask = torch.cat([pad(m.long()).bool() for m in self._mask])
        label, ordinal = torch.cat(self._label), torch.cat(self._ordinal)
        sources = self._source
        by_source = {}
        for s in sorted(set(sources)):
            sel = torch.tensor([x == s for x in sources])
            by_source[s] = compute_metrics(probs[sel], label[sel], mask[sel], ordinal[sel])
        return {"overall": compute_metrics(probs, label, mask, ordinal), "by_source": by_source}
