"""End-to-end: Lightning training -> checkpoint -> plain PyTorch evaluation."""

import json
from pathlib import Path

import pytest
import yaml

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


@pytest.mark.parametrize("raw", ["1e-3", "1e3", "1.0e3", ".5e-2", "-2E+3"])
def test_scientific_notation_in_config_and_overrides(tmp_path, raw):
    path = tmp_path / "c.yaml"
    path.write_text(f"optim: {{lr: {raw}}}\nquoted: '{raw}'\n")
    cfg = load_config(path, [f"override={raw}", f"quoted_override='{raw}'"])
    assert isinstance(cfg["optim"]["lr"], float)
    assert cfg["optim"]["lr"] == cfg["override"] == float(raw)
    assert cfg["quoted"] == cfg["quoted_override"] == raw


def test_config_loader_keeps_safe_yaml_semantics(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("{}\n")
    with pytest.raises(yaml.constructor.ConstructorError):
        load_config(path, ["value=!!python/object:builtins.object {}"])
    assert yaml.safe_load("1e-3") == "1e-3"  # custom resolution does not mutate SafeLoader


def test_train_without_logging(tmp_path):
    cfg = load_config("configs/smoke.yaml", [
        f"trainer.default_root_dir={tmp_path / 'run'}", "log=false", "optim.lr=1e-3",
        "trainer.max_epochs=1", "trainer.enable_progress_bar=false", "trainer.enable_model_summary=false",
        "data.synthetic_train=16", "data.synthetic_val=8", "data.batch_size=8",
    ])
    best = train(cfg)
    assert Path(best).is_file()
    assert not (tmp_path / "run" / "logs").exists()


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
