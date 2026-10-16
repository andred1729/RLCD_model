"""Pair tokenizers: encode ``(prompt, option)`` into one sequence per option.

Every option is scored independently, so the tokenizer only ever sees one
(prompt, option) pair at a time. ``ByteTokenizer`` needs no downloads and is used
for tests and smoke runs; ``HFTokenizer`` wraps a Hugging Face tokenizer.
"""

from __future__ import annotations

from typing import Protocol

import torch


class PairTokenizer(Protocol):
    pad_id: int
    vocab_size: int

    def encode_pair(self, prompt: str, option: str, max_length: int) -> list[int]: ...


class ByteTokenizer:
    """UTF-8 bytes + 3 specials. Layout: ``[BOS] prompt [SEP] option``."""

    pad_id = 0
    bos_id = 1
    sep_id = 2
    _offset = 3
    vocab_size = 256 + 3

    def encode_pair(self, prompt: str, option: str, max_length: int) -> list[int]:
        if max_length < 4:
            raise ValueError("max_length must be >= 4")
        budget = max_length - 2  # BOS + SEP
        prm = [b + self._offset for b in prompt.encode("utf-8")]
        # Reserve half the budget for options, plus any space a short prompt does not need.
        option_budget = max(budget // 2, budget - len(prm))
        opt = [b + self._offset for b in option.encode("utf-8")][:option_budget]
        room = budget - len(opt)
        if len(prm) > room:  # keep the tail of the prompt: questions usually come last
            prm = prm[len(prm) - room :]
        return [self.bos_id, *prm, self.sep_id, *opt]


class HFTokenizer:
    def __init__(self, name_or_path: str):
        from transformers import AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(name_or_path)
        if self._tok.pad_token_id is None:
            self._tok.pad_token = self._tok.eos_token
        self.pad_id = int(self._tok.pad_token_id)
        self.vocab_size = len(self._tok)

    def encode_pair(self, prompt: str, option: str, max_length: int) -> list[int]:
        out = self._tok(prompt, option, truncation="longest_first", max_length=max_length)
        return list(out["input_ids"])


def build_tokenizer(spec: str) -> PairTokenizer:
    """``"byte"`` or ``"hf:<name_or_path>"``."""
    if spec == "byte":
        return ByteTokenizer()
    if spec.startswith("hf:"):
        return HFTokenizer(spec[3:])
    raise ValueError(f"unknown tokenizer spec {spec!r}; use 'byte' or 'hf:<name_or_path>'")


def encode_batch(
    tokenizer: PairTokenizer,
    prompts: list[str],
    options: list[list[str]],
    max_length: int,
) -> dict[str, torch.Tensor]:
    """Encode a batch into ``input_ids``/``attention_mask`` of shape (B, K, L) and ``option_mask`` (B, K)."""
    batch, k_max = len(prompts), max(len(o) for o in options)
    rows = [[tokenizer.encode_pair(p, o, max_length) for o in opts] for p, opts in zip(prompts, options)]
    length = max(len(seq) for row in rows for seq in row)
    input_ids = torch.full((batch, k_max, length), tokenizer.pad_id, dtype=torch.long)
    attention_mask = torch.zeros((batch, k_max, length), dtype=torch.long)
    option_mask = torch.zeros((batch, k_max), dtype=torch.bool)
    for i, row in enumerate(rows):
        for j, seq in enumerate(row):
            input_ids[i, j, : len(seq)] = torch.tensor(seq)
            attention_mask[i, j, : len(seq)] = 1
            option_mask[i, j] = True
    return {"input_ids": input_ids, "attention_mask": attention_mask, "option_mask": option_mask}
