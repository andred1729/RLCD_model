"""Evaluation in plain PyTorch (no Lightning in the loop).

    python -m rlcd.evaluate --checkpoint outputs/logs/version_0/checkpoints/best.ckpt \\
        --data val.jsonl --output results.json

Reports accuracy, NLL, Brier, ECE (and RPS on ordinal rows), overall and per ``source``, plus an
option-order sensitivity check: each example is re-scored with shuffled options and we report the
fraction of examples whose chosen option changes (0.0 for order-invariant models).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import Collator, Example, ExampleDataset, load_jsonl, move_batch
from .metrics import MetricAccumulator
from .model import ModelConfig, OptionScorer, build_model
from .tokenizer import PairTokenizer


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[OptionScorer, PairTokenizer, ModelConfig]:
    """Rebuild a model from a Lightning checkpoint without instantiating a LightningModule."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    cfg = ModelConfig.from_dict(ckpt["hyper_parameters"]["model_cfg"])
    model, tokenizer = build_model(cfg)
    state = {k.removeprefix("model."): v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
    model.load_state_dict(state)
    return model.to(device).eval(), tokenizer, cfg


@torch.no_grad()
def predict_probs(
    model: OptionScorer,
    examples: list[Example],
    tokenizer: PairTokenizer,
    max_length: int,
    batch_size: int = 64,
    device: str = "cpu",
) -> list[torch.Tensor]:
    """Per-example probability vectors (length = that example's number of options)."""
    loader = DataLoader(ExampleDataset(examples), batch_size, collate_fn=Collator(tokenizer, max_length))
    out: list[torch.Tensor] = []
    for batch in loader:
        batch = move_batch(batch, device)
        probs = model(batch["input_ids"], batch["attention_mask"], batch["option_mask"]).softmax(-1).cpu()
        sizes = batch["option_mask"].sum(-1).tolist()
        out += [p[:n] for p, n in zip(probs, sizes)]
    return out


def order_sensitivity(
    model: OptionScorer,
    examples: list[Example],
    tokenizer: PairTokenizer,
    max_length: int,
    base_probs: list[torch.Tensor],
    batch_size: int = 64,
    device: str = "cpu",
    seed: int = 0,
) -> dict[str, float]:
    """Shuffle options of non-ordinal examples (ordinal order carries meaning) and compare."""
    rng = random.Random(seed)
    keep, perms, shuffled = [], [], []
    for i, ex in enumerate(examples):
        if ex.ordinal:
            continue
        perm = list(range(len(ex.options)))
        rng.shuffle(perm)
        keep.append(i)
        perms.append(perm)
        options = [ex.options[j] for j in perm]
        shuffled.append(Example(ex.prompt, options, perm.index(ex.label), False, ex.source))
    if not shuffled:
        return {"n": 0, "flip_rate": 0.0, "mean_abs_prob_delta": 0.0}
    new = predict_probs(model, shuffled, tokenizer, max_length, batch_size, device)
    flips, deltas = 0, []
    for i, perm, p_new in zip(keep, perms, new):
        restored = torch.zeros_like(base_probs[i])
        restored[torch.tensor(perm)] = p_new  # shuffled slot s holds original option perm[s]
        chosen = perm[p_new.argmax().item()]
        flips += int(chosen != base_probs[i].argmax().item())
        deltas.append((restored - base_probs[i]).abs().mean().item())
    return {"n": len(keep), "flip_rate": flips / len(keep), "mean_abs_prob_delta": sum(deltas) / len(deltas)}


def evaluate(
    model: OptionScorer,
    examples: list[Example],
    tokenizer: PairTokenizer,
    max_length: int,
    batch_size: int = 64,
    device: str = "cpu",
    check_order: bool = True,
    seed: int = 0,
) -> dict:
    model.eval()
    probs = predict_probs(model, examples, tokenizer, max_length, batch_size, device)
    acc = MetricAccumulator()
    k = max(len(ex.options) for ex in examples)
    for ex, p in zip(examples, probs):
        padded = torch.zeros(1, k)
        padded[0, : len(p)] = p
        mask = torch.zeros(1, k, dtype=torch.bool)
        mask[0, : len(p)] = True
        acc.update(padded, torch.tensor([ex.label]), mask, torch.tensor([ex.ordinal]), [ex.source])
    result = acc.compute()
    if check_order:
        result["order_sensitivity"] = order_sensitivity(
            model, examples, tokenizer, max_length, probs, batch_size, device, seed
        )
    return result


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True, help="JSONL file of decision examples")
    p.add_argument("--output", help="write results JSON here")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-order-check", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    model, tokenizer, cfg = load_checkpoint(args.checkpoint, args.device)
    result = evaluate(
        model, load_jsonl(args.data), tokenizer, cfg.max_length, args.batch_size, args.device,
        check_order=not args.no_order_check, seed=args.seed,
    )
    text = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
