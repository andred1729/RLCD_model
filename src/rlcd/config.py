"""YAML config loading with ``a.b.c=value`` command-line overrides."""

from __future__ import annotations

import re
from pathlib import Path

import yaml


class _ConfigLoader(yaml.SafeLoader):
    """Safe YAML loading with scientific notation such as ``1e-3`` recognised as a float."""


_ConfigLoader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(r"^[-+]?(?:[0-9][0-9_]*(?:\.[0-9_]*)?|\.[0-9_]+)[eE][-+]?[0-9]+$"),
    list("-+0123456789."),
)


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=_ConfigLoader) or {}
    for item in overrides or []:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"override {item!r} must look like section.key=value")
        node = cfg
        *parents, leaf = key.split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = yaml.load(raw, Loader=_ConfigLoader)
    return cfg
