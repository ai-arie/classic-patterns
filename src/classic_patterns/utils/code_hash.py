"""Code version identifier recorded in data manifests and experiment runs.

Uses the git commit when the project is a git repository with a clean tree,
otherwise a SHA-256 over all files in ``src/``, ``scripts/`` and ``configs/``.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

HASHED_DIRS = ("src", "scripts", "configs")
SKIP_PARTS = {"__pycache__", ".pytest_cache", ".ruff_cache"}


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _git_commit(root: Path) -> str | None:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--", *HASHED_DIRS],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return f"git:{commit}" + ("-dirty" if dirty else "")


def tree_hash(root: Path) -> str:
    h = hashlib.sha256()
    for d in HASHED_DIRS:
        base = root / d
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file() or SKIP_PARTS.intersection(path.parts) or path.suffix == ".pyc":
                continue
            h.update(path.relative_to(root).as_posix().encode())
            h.update(b"\0")
            h.update(path.read_bytes())
            h.update(b"\0")
    return f"sha256:{h.hexdigest()}"


def code_hash(root: Path | None = None) -> str:
    root = root or project_root()
    return _git_commit(root) or tree_hash(root)
