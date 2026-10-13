"""YAML config loading with ``a.b.c=value`` command-line overrides."""

from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    for item in overrides or []:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"override {item!r} must look like section.key=value")
        node = cfg
        *parents, leaf = key.split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = yaml.safe_load(raw)
    return cfg
