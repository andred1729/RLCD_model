"""End-to-end: Lightning training -> checkpoint -> plain PyTorch evaluation."""

import json

from rlcd.config import load_config
from rlcd.data import save_jsonl
from rlcd.evaluate import evaluate, load_checkpoint
from rlcd.evaluate import main as eval_main
from rlcd.synthetic import make_synthetic
from rlcd.train import train


def test_override_parsing(tmp_path):
    (tmp_path / "c.yaml").write_text("a: {b: 1}\n")
    cfg = load_config(tmp_path / "c.yaml", ["a.b=2", "c.d.e=[1, 2]", "f=null"])
    assert cfg == {"a": {"b": 2}, "c": {"d": {"e": [1, 2]}}, "f": None}


def test_train_then_evaluate(tmp_path):
    cfg = load_config("configs/smoke.yaml", [
        f"trainer.default_root_dir={tmp_path / 'run'}", "trainer.max_epochs=6",
        "trainer.enable_progress_bar=false", "data.synthetic_train=1024", "data.synthetic_val=128",
    ])
    best = train(cfg)
    assert best.endswith(".ckpt")

    model, tok, mcfg = load_checkpoint(best)
    examples = make_synthetic(200, seed=99)
    result = evaluate(model, examples, tok, mcfg.max_length)
    assert result["overall"]["accuracy"] > 0.9  # chance is ~0.3
    assert result["order_sensitivity"]["flip_rate"] == 0.0
    assert set(result["by_source"]) == {"match", "rating"}

    data, out = tmp_path / "d.jsonl", tmp_path / "r.json"
    save_jsonl(examples[:50], data)
    eval_main(["--checkpoint", best, "--data", str(data), "--output", str(out), "--device", "cpu"])
    assert json.loads(out.read_text())["overall"]["n"] == 50
