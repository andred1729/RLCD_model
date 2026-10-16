"""Tiny synthetic decision tasks for smoke tests and pipeline debugging.

* ``match``: pick the option equal to the key in the prompt (variable option count).
* ``rating``: ordinal; the prompt contains a digit d, pick option "d stars" out of 5.
"""

from __future__ import annotations

import argparse
import random

from .data import Example, save_jsonl

_LETTERS = "abcdefgh"


def make_synthetic(n: int, seed: int = 0) -> list[Example]:
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        if rng.random() < 0.5:
            k = rng.randint(2, 5)
            options = rng.sample(_LETTERS, k)
            label = rng.randrange(k)
            out.append(Example(f"key: {options[label]}", options, label, False, "match"))
        else:
            d = rng.randint(1, 5)
            options = [f"{i} stars" for i in range(1, 6)]
            out.append(Example(f"rating digit: {d}", options, d - 1, True, "rating"))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Write a synthetic JSONL dataset.")
    p.add_argument("output")
    p.add_argument("-n", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    save_jsonl(make_synthetic(args.n, args.seed), args.output)
    print(f"wrote {args.n} examples to {args.output}")


if __name__ == "__main__":
    main()
