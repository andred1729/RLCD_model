"""Decision examples: JSONL IO, collation, and the Lightning DataModule.

JSONL schema, one object per line::

    {"prompt": str, "options": [str, ...], "label": int,
     "ordinal": bool (optional, options are ordered low -> high),
     "source": str (optional, used for per-source eval breakdowns)}
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset

from .tokenizer import PairTokenizer, build_tokenizer, encode_batch


@dataclass
class Example:
    prompt: str
    options: list[str]
    label: int
    ordinal: bool = False
    source: str = "default"

    def __post_init__(self) -> None:
        if len(self.options) < 2:
            raise ValueError(f"need at least 2 options, got {len(self.options)}")
        if not 0 <= self.label < len(self.options):
            raise ValueError(f"label {self.label} out of range for {len(self.options)} options")


def load_jsonl(path: str | Path) -> list[Example]:
    examples = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                examples.append(Example(**json.loads(line)))
            except (TypeError, ValueError) as e:
                raise ValueError(f"{path}:{lineno}: {e}") from e
    return examples


def save_jsonl(examples: Sequence[Example], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(asdict(ex), ensure_ascii=False) + "\n")


class ExampleDataset(Dataset):
    def __init__(self, examples: Sequence[Example]):
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Example:
        return self.examples[idx]


class Collator:
    """Tokenizes in the DataLoader workers. Picklable."""

    def __init__(self, tokenizer: PairTokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, examples: list[Example]) -> dict:
        batch = encode_batch(
            self.tokenizer, [e.prompt for e in examples], [e.options for e in examples], self.max_length
        )
        batch["label"] = torch.tensor([e.label for e in examples], dtype=torch.long)
        batch["ordinal"] = torch.tensor([e.ordinal for e in examples], dtype=torch.bool)
        batch["source"] = [e.source for e in examples]
        return batch


def move_batch(batch: dict, device: torch.device | str) -> dict:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


class RLCDDataModule(L.LightningDataModule):
    """Loads JSONL files, or generates synthetic data when a path is ``None``."""

    def __init__(
        self,
        tokenizer: str = "byte",
        max_length: int = 128,
        train_path: str | None = None,
        val_path: str | None = None,
        test_path: str | None = None,
        synthetic_train: int = 2048,
        synthetic_val: int = 256,
        batch_size: int = 32,
        num_workers: int = 0,
        seed: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer = build_tokenizer(tokenizer)
        self.collator = Collator(self.tokenizer, max_length)
        self.train_set = self.val_set = self.test_set = None

    def setup(self, stage: str | None = None) -> None:
        from .synthetic import make_synthetic

        hp = self.hparams
        if stage in (None, "fit"):
            self.train_set = ExampleDataset(
                load_jsonl(hp.train_path) if hp.train_path else make_synthetic(hp.synthetic_train, hp.seed)
            )
            self.val_set = ExampleDataset(
                load_jsonl(hp.val_path) if hp.val_path else make_synthetic(hp.synthetic_val, hp.seed + 1)
            )
        if stage in (None, "test") and hp.test_path:
            self.test_set = ExampleDataset(load_jsonl(hp.test_path))

    def _loader(self, ds: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.hparams.batch_size,
            shuffle=shuffle,
            num_workers=self.hparams.num_workers,
            collate_fn=self.collator,
            persistent_workers=self.hparams.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_set, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_set, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_set, shuffle=False)
