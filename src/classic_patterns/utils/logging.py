"""Logging setup: console plus an optional log file for long jobs."""

from __future__ import annotations

import logging
from pathlib import Path

FMT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def setup_logging(log_file: str | Path | None = None, level: int = logging.INFO) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=level, format=FMT, handlers=handlers, force=True)
    # yfinance logs every failed ticker at ERROR; we record those ourselves.
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
