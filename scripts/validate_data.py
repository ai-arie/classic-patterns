"""Validate the raw downloaded prices and decide the cleaning actions.

python scripts/validate_data.py

Writes artifacts/validation/{summary,events}.parquet and data/meta/exclusions.parquet.
"""

from __future__ import annotations

import argparse
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import pandas as pd

from classic_patterns.data.calendar import nyse_sessions
from classic_patterns.data.validate import ACTION_COLS, check_ticker, decide
from classic_patterns.utils.code_hash import code_hash
from classic_patterns.utils.config import load_config, resolve
from classic_patterns.utils.logging import setup_logging

log = logging.getLogger("validate_data")


def _check_file(path: Path, sessions: pd.DatetimeIndex, cfg: dict, is_benchmark: bool):
    df = pd.read_parquet(path)
    summary, events = check_ticker(df, sessions, cfg["checks"], is_benchmark)
    return summary, events, decide(df, events, cfg["cleaning"], is_benchmark)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="validate")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths = {k: resolve(v) for k, v in cfg["paths"].items()}
    setup_logging(paths["log_file"])
    log.info("code %s", code_hash())

    sessions = nyse_sessions("1900-01-01", pd.Timestamp.today())
    jobs = [(p, False) for p in sorted(paths["raw_prices"].glob("*.parquet"))]
    jobs += [(p, True) for p in sorted(paths["raw_benchmarks"].glob("*.parquet"))]
    summaries, events, actions = [], [], []
    with ProcessPoolExecutor(args.workers) as ex:
        for is_bm in (False, True):
            files = [p for p, b in jobs if b == is_bm]
            fn = partial(_check_file, sessions=sessions, cfg=cfg, is_benchmark=is_bm)
            for s, e, a in ex.map(fn, files, chunksize=32):
                summaries.append(s)
                events += e
                actions += a
    summary = pd.DataFrame(summaries)
    ev = pd.DataFrame(
        events, columns=["ticker", "date", "end", "kind", "value", "detail", "relevant"]
    )
    excl = pd.DataFrame(actions, columns=ACTION_COLS)
    paths["out_dir"].mkdir(parents=True, exist_ok=True)
    summary.to_parquet(paths["out_dir"] / "summary.parquet", index=False)
    ev.to_parquet(paths["out_dir"] / "events.parquet", index=False)
    excl.to_parquet(paths["exclusions"], compression="zstd", index=False)
    log.info(
        "%d tickers, %d events: %s", len(summary), len(ev), ev["kind"].value_counts().to_dict()
    )
    log.info("actions: %s", excl.groupby(["action", "reason"]).size().to_dict())


if __name__ == "__main__":
    main()
