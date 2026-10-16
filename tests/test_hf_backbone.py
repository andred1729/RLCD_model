"""HF backbone + LoRA, using a tiny randomly initialised BERT built locally (no network)."""

import pytest
import torch

transformers = pytest.importorskip("transformers")
pytest.importorskip("peft")

from rlcd.data import Collator  # noqa: E402
from rlcd.losses import rlcd_loss  # noqa: E402
from rlcd.model import ModelConfig, build_model  # noqa: E402
from rlcd.synthetic import make_synthetic  # noqa: E402


@pytest.fixture()
def tiny_bert(tmp_path):
    specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    vocab = [*specials, *"abcdefgh12345:. ", "key", "stars", "rating", "digit"]
    vocab = list(dict.fromkeys(vocab))
    (tmp_path / "vocab.txt").write_text("\n".join(vocab))
    tok = transformers.BertTokenizerFast(str(tmp_path / "vocab.txt"))
    tok.save_pretrained(tmp_path)
    cfg = transformers.BertConfig(
        vocab_size=len(vocab), hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
        intermediate_size=64, max_position_embeddings=64,
    )
    transformers.BertModel(cfg).save_pretrained(tmp_path)
    return str(tmp_path)


def test_lora_trains_only_adapters_and_head(tiny_bert):
    cfg = ModelConfig(
        tokenizer=f"hf:{tiny_bert}", backbone=f"hf:{tiny_bert}", max_length=48, pooling="cls",
        lora={"r": 4, "target_modules": ["query", "value"]},
    )
    model, tok = build_model(cfg)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable and all("lora_" in n or n.startswith("head") for n in trainable)

    b = Collator(tok, 48)(make_synthetic(8, seed=0))
    logits = model(b["input_ids"], b["attention_mask"], b["option_mask"])
    loss, _ = rlcd_loss(logits, b["label"], b["option_mask"], b["ordinal"])
    loss.backward()
    lora_grads = [p.grad for n, p in model.named_parameters() if "lora_" in n]
    assert any(g is not None and g.abs().sum() > 0 for g in lora_grads)
    assert torch.isfinite(loss)
