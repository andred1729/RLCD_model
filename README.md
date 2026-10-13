# RLCD_model

Training and evaluation scaffold for **System One** style decision models: given a prompt and a set
of options, output a calibrated probability for each option (no text generation).

* **Training:** PyTorch Lightning (`rlcd.train`)
* **Evaluation:** plain PyTorch, no Lightning in the loop (`rlcd.evaluate`)
* **Objective:** RLCD-style proper scoring reward (log + spherical, plus ranked probability score for ordinal questions)

## Install

```bash
pip install -e ".[dev]"        # core + tests
pip install -e ".[dev,hf]"     # + transformers and peft for Hugging Face backbones / LoRA
```

## Quickstart (CPU, about a minute)

```bash
rlcd-train --config configs/smoke.yaml                      # trains on synthetic data
rlcd-synth /tmp/val.jsonl -n 300 --seed 123                 # make an eval set
rlcd-eval --checkpoint outputs/smoke/logs/version_0/checkpoints/best-*.ckpt --data /tmp/val.jsonl
pytest
```

Override any config value from the command line: `rlcd-train --config configs/smoke.yaml --set trainer.max_epochs=2 optim.lr=1e-3`.

## Data format

JSONL, one decision per line:

```json
{"prompt": "Which city is the capital of France?", "options": ["Lyon", "Paris", "Nice"], "label": 1,
 "ordinal": false, "source": "geography"}
```

* `label` is the index of the correct option.
* `ordinal: true` means options are ordered low to high (ratings); the ranked probability score then penalises distant mistakes more.
* `source` is optional and gives per-source metric breakdowns.

## How it works

| Piece | File | Notes |
|---|---|---|
| Option scorer | `src/rlcd/model.py` | Each `(prompt, option)` is encoded independently by a shared backbone, pooled, and mapped to a scalar; softmax over the real options. Options never see each other, so the model is order-equivariant by construction. |
| Backbones | `src/rlcd/backbones.py` | `toy` (from-scratch transformer, no downloads) or `hf:<name_or_path>` with optional freezing and LoRA. |
| Loss | `src/rlcd/losses.py` | `reward = w_log*log p_y + w_sph*p_y/‖p‖ − w_rps*RPS`. A positive mix of proper scoring rules is proper, so calibration is rewarded. The loss is `−reward`, so its floor is `−w_sph` (−0.5 by default), not 0. |
| Lightning module | `src/rlcd/lit.py` | AdamW, warmup + cosine schedule, val loss/acc/NLL/Brier/ECE. |
| Evaluation | `src/rlcd/evaluate.py`, `metrics.py` | Accuracy, NLL, Brier, ECE, ordinal RPS, per-source breakdown, and an option-shuffle check reporting how often the chosen option changes. |

Not included yet: the packed shared-prefix layout (one sequence with a block attention mask), the
context resampler and recurrent scoring head from the reference design, dataset builders, and
multi-GPU metric syncing for the epoch-level calibration metrics. `configs/default.yaml` is an
untested template for a LoRA run on a Hugging Face encoder; the Hugging Face path is only covered
by a test with a tiny locally built BERT.

## Layout

```
configs/      smoke.yaml (CPU, synthetic), default.yaml (HF + LoRA template)
src/rlcd/     tokenizer, data, backbones, model, losses, metrics, lit, train, evaluate, synthetic
tests/        losses (propriety), data, model (order equivariance, padding), metrics, e2e pipeline, HF+LoRA
```
