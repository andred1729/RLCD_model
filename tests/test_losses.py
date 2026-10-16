import torch

from rlcd.losses import ScoringWeights, rlcd_loss, scoring_reward


def _expected_reward(p, q, ordinal):
    """E_{y~q} reward(p, y) for a single prediction p over len(q) options."""
    k = len(q)
    logits = p.log().expand(k, k)
    out = scoring_reward(
        logits, torch.arange(k), torch.ones(k, k, dtype=torch.bool), torch.full((k,), ordinal)
    )
    return (q * out["reward"]).sum()


def test_scoring_rule_is_proper():
    # Expected reward under truth q must be maximal when the report equals q.
    torch.manual_seed(0)
    for ordinal in (False, True):
        q = torch.tensor([0.2, 0.5, 0.3])
        best = _expected_reward(q, q, ordinal)
        for _ in range(500):
            p = torch.distributions.Dirichlet(torch.ones(3)).sample()
            assert _expected_reward(p, q, ordinal) <= best + 1e-6


def test_perfect_prediction_reward():
    logits = torch.tensor([[50.0, -50.0, -50.0]])
    out = scoring_reward(
        logits, torch.tensor([0]), torch.ones(1, 3, dtype=torch.bool), torch.tensor([True])
    )
    w = ScoringWeights()
    assert torch.allclose(out["reward"], torch.tensor([w.spherical]), atol=1e-4)
    assert out["rps"].item() < 1e-6


def test_rps_penalises_distance_on_ordinal_only():
    mask = torch.ones(1, 5, dtype=torch.bool)
    near = torch.tensor([[0.0, 5.0, 0.0, 0.0, 0.0]])  # predicts 1, truth 0
    far = torch.tensor([[0.0, 0.0, 0.0, 0.0, 5.0]])  # predicts 4, truth 0
    lab = torch.tensor([0])
    ord_near = scoring_reward(near, lab, mask, torch.tensor([True]))["rps"]
    ord_far = scoring_reward(far, lab, mask, torch.tensor([True]))["rps"]
    assert ord_far > ord_near > 0
    assert scoring_reward(far, lab, mask, torch.tensor([False]))["rps"].item() == 0.0


def test_padded_options_get_no_mass_and_finite_grads():
    logits = torch.tensor([[1.0, 2.0, float("-inf")], [0.5, 0.1, 0.2]], requires_grad=True)
    mask = torch.tensor([[True, True, False], [True, True, True]])
    loss, out = rlcd_loss(logits, torch.tensor([1, 2]), mask, torch.tensor([True, True]))
    assert out["probs"][0, 2] == 0
    loss.backward()
    assert torch.isfinite(logits.grad[mask]).all()
