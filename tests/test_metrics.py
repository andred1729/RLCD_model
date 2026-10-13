import torch

from rlcd.metrics import MetricAccumulator, compute_metrics


def test_perfect_and_uniform():
    mask = torch.ones(4, 4, dtype=torch.bool)
    label = torch.tensor([0, 1, 2, 3])
    perfect = compute_metrics(torch.eye(4), label, mask, torch.zeros(4, dtype=torch.bool))
    assert perfect["accuracy"] == 1.0 and perfect["brier"] == 0.0 and perfect["ece"] == 0.0
    uniform = compute_metrics(torch.full((4, 4), 0.25), label, mask, torch.zeros(4, dtype=torch.bool))
    assert abs(uniform["nll"] - torch.tensor(4.0).log().item()) < 1e-6
    assert abs(uniform["brier"] - 0.75) < 1e-6


def test_ece_detects_overconfidence():
    probs = torch.tensor([[0.99, 0.01]] * 10)
    label = torch.tensor([0] * 5 + [1] * 5)  # right half the time while claiming 99%
    m = compute_metrics(probs, label, torch.ones(10, 2, dtype=torch.bool), torch.zeros(10, dtype=torch.bool))
    assert abs(m["ece"] - 0.49) < 1e-5


def test_accumulator_mixed_widths_and_sources():
    acc = MetricAccumulator()
    acc.update(torch.tensor([[1.0, 0.0]]), torch.tensor([0]), torch.tensor([[True, True]]),
               torch.tensor([False]), ["a"])
    acc.update(torch.tensor([[0.0, 0.0, 1.0]]), torch.tensor([1]), torch.ones(1, 3, dtype=torch.bool),
               torch.tensor([True]), ["b"])
    r = acc.compute()
    assert r["overall"]["n"] == 2 and r["overall"]["accuracy"] == 0.5
    assert r["by_source"]["a"]["accuracy"] == 1.0 and r["by_source"]["b"]["accuracy"] == 0.0
    assert "rps_ordinal" in r["by_source"]["b"]
