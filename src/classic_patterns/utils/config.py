"""YAML config loading and the local ``.env`` file."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from classic_patterns.utils.code_hash import project_root


def load_config(name_or_path: str | Path) -> dict[str, Any]:
    """Load ``configs/<name>.yaml`` or an explicit path."""
    path = Path(name_or_path)
    if path.suffix not in (".yaml", ".yml"):
        path = project_root() / "configs" / f"{name_or_path}.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def resolve(path: str | Path) -> Path:
    """Resolve a config path relative to the project root."""
    p = Path(path)
    return p if p.is_absolute() else project_root() / p


def load_env(path: str | Path | None = None) -> dict[str, str]:
    """Read ``KEY=VALUE`` lines from ``<project>/.env`` (local, never committed; template:
    ``.env.example``) into ``os.environ`` without overriding variables that are already set.
    Blank lines and ``#`` comments are skipped; surrounding quotes are stripped. Returns the
    values read (empty if the file does not exist)."""
    path = Path(path) if path is not None else project_root() / ".env"
    if not path.is_file():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.removeprefix("export ").split("=", 1)
        key, val = key.strip(), val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        out[key] = val
        os.environ.setdefault(key, val)
    return out
