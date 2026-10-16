import torch

from rlcd.data import Collator
from rlcd.model import ModelConfig, build_model
from rlcd.synthetic import make_synthetic


def _batch(n=8, max_length=48):
    model, tok = build_model(ModelConfig(max_length=max_length, head_dropout=0.0))
    return model.eval(), Collator(tok, max_length)(make_synthetic(n, seed=3))


def test_forward_shape_and_masking():
    model, b = _batch()
    logits = model(b["input_ids"], b["attention_mask"], b["option_mask"])
    assert logits.shape == b["option_mask"].shape
    assert torch.isinf(logits[~b["option_mask"]]).all()
    assert torch.isfinite(logits[b["option_mask"]]).all()


def test_equivariant_to_option_order():
    model, b = _batch()
    base = model(b["input_ids"], b["attention_mask"], b["option_mask"])
    k = b["option_mask"].shape[1]
    perm = torch.tensor([(i + 1) % k for i in range(k)])
    # Rotate slots; the padding pattern changes, so compare only rows with all K options.
    full = b["option_mask"].all(-1)
    out = model(b["input_ids"][:, perm], b["attention_mask"][:, perm], b["option_mask"][:, perm])
    assert full.any()
    assert torch.allclose(out[full], base[full][:, perm], atol=1e-5)


def test_padding_does_not_change_scores():
    model, tok = build_model(ModelConfig(max_length=48, head_dropout=0.0))
    model.eval()
    ex = make_synthetic(1, seed=5)
    solo = Collator(tok, 48)(ex)
    longer = make_synthetic(40, seed=6)
    mixed = Collator(tok, 48)([ex[0], *longer])
    a = model(solo["input_ids"], solo["attention_mask"], solo["option_mask"])[0]
    b = model(mixed["input_ids"], mixed["attention_mask"], mixed["option_mask"])[0]
    k = int(solo["option_mask"].sum())
    assert torch.allclose(a[:k], b[:k], atol=1e-5)


def test_pooling_modes():
    from rlcd.backbones import pool

    h = torch.arange(24.0).reshape(2, 4, 3)
    m = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
    assert torch.allclose(pool(h, m, "mean")[1], h[1, :2].mean(0))
    assert torch.equal(pool(h, m, "last")[0], h[0, 2])
    assert torch.equal(pool(h, m, "cls")[1], h[1, 0])
