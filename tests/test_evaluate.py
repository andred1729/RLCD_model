import pytest
import torch

from rlcd.data import Example
from rlcd.evaluate import order_sensitivity
from rlcd.tokenizer import ByteTokenizer


class FixedSlotScorer(torch.nn.Module):
    """A position-dependent scorer for exercising shuffle diagnostics."""

    def __init__(self, probabilities):
        super().__init__()
        self.logits = torch.tensor(probabilities).log()

    def forward(self, input_ids, attention_mask, option_mask):
        return self.logits.expand(input_ids.shape[0], -1)


def test_order_sensitivity_counts_tie_breaking_choice_changes():
    probs = [0.5, 0.5]
    result = order_sensitivity(
        FixedSlotScorer(probs), [Example("q", ["a", "b"], 0)], ByteTokenizer(), 64,
        [torch.tensor(probs)], seed=1,  # swaps the two options
    )
    assert result["flip_rate"] == 1.0
    assert result["mean_abs_prob_delta"] == 0.0


def test_order_sensitivity_reports_mean_absolute_probability_delta():
    probs = [0.6, 0.3, 0.1]
    result = order_sensitivity(
        FixedSlotScorer(probs), [Example("q", ["a", "b", "c"], 0)], ByteTokenizer(), 64,
        [torch.tensor(probs)], seed=1,  # shuffled slots correspond to original options [1, 2, 0]
    )
    assert result["flip_rate"] == 1.0
    assert result["mean_abs_prob_delta"] == pytest.approx((0.5 + 0.3 + 0.2) / 3)
