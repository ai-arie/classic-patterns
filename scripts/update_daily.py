"""Bring stored daily prices up to the last completed NYSE session. Idempotent.

For the screener app. Training and evaluation use the frozen snapshot from the full download.

python scripts/update_daily.py                     # universe + benchmarks + macro
python scripts/update_daily.py --test              # the 20 test tickers only
python scripts/update_daily.py --test --raw-dir D  # operate on another copy of data/raw
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

from classic_patterns.data.calendar import last_completed_session
from classic_patterns.data.download import DownloadState, StopDownload, download_fred, now_iso
from classic_patterns.data.update import Updater, refresh_universe
from classic_patterns.utils.code_hash import code_hash
from classic_patterns.utils.config import load_config, resolve
from classic_patterns.utils.logging import setup_logging

log = logging.getLogger("update_daily")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--test", action="store_true", help="update the test tickers only")
    g.add_argument("--symbols", nargs="+", help="update these Yahoo symbols only")
    ap.add_argument("--raw-dir", type=Path, help="use this directory instead of data/raw")
    ap.add_argument("--refresh-universe", action="store_true", help="refresh even if not due")
    ap.add_argument("--no-macro", action="store_true", help="skip FRED series")
    ap.add_argument("--config", default="download")
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths = {k: resolve(v) for k, v in cfg["paths"].items()}
    if args.raw_dir is not None:
        raw = args.raw_dir.resolve()
        for key, sub in (
            ("raw_prices", "prices"),
            ("raw_benchmarks", "benchmarks"),
            ("raw_macro", "macro"),
        ):
            paths[key] = raw / sub
        paths["state_file"] = raw / "download_state.json"
        paths["update_state_file"] = raw / "update_state.json"
    setup_logging(paths["update_log_file"])
    log.info("yfinance %s, pandas %s, code %s", yf.__version__, pd.__version__, code_hash())

    ustate_path = paths["update_state_file"]
    ustate = json.loads(ustate_path.read_text()) if ustate_path.exists() else {}
    last_session = last_completed_session(settle_minutes=cfg["update"]["settle_minutes"])

    if args.test:
        symbols = list(cfg["test_tickers"])
    elif args.symbols:
        symbols = args.symbols
    else:
        last_refresh = ustate.get("universe_refreshed_at")
        due = last_refresh is None or pd.Timestamp.now(tz="UTC") - pd.Timestamp(
            last_refresh
        ) >= pd.Timedelta(days=cfg["update"]["universe_refresh_days"])
        if due or args.refresh_universe:
            refresh_universe(load_config("universe"), paths["universe"], paths["universe_changes"])
            ustate["universe_refreshed_at"] = now_iso()
        universe = pd.read_parquet(paths["universe"])
        symbols = list(cfg["benchmarks"]) + universe["yahoo_symbol"].tolist()

    if not args.no_macro:
        for series, spec in cfg["fred"]["series"].items():
            try:
                df = download_fred(series, cfg["fred"]["url"], paths["raw_macro"])
            except requests.RequestException as e:
                # FRED down: keep the stored series (macro inputs move slowly) instead of
                # stopping the whole price update.
                stored = paths["raw_macro"] / f"{series}.parquet"
                if stored.exists():
                    last = pd.read_parquet(stored, columns=["date"])["date"].max().date()
                    log.warning(
                        "FRED %s unavailable (%s); keeping stored data to %s", series, e, last
                    )
                elif (spec or {}).get("optional"):
                    log.warning("FRED %s unavailable (%s); optional, no stored data", series, e)
                else:
                    raise
                continue
            log.info("FRED %s: last obs %s", series, df["date"].max().date())

    state = DownloadState.load(paths["state_file"])
    dirs = {"prices": paths["raw_prices"], "benchmarks": paths["raw_benchmarks"]}
    try:
        counts = Updater(cfg, state, dirs, last_session).run(symbols)
    except StopDownload as e:
        state.save()
        log.error("stopped cleanly: %s. Re-run the same command later.", e)
        return 2

    ustate.update(
        {"last_run_at": now_iso(), "last_session": str(last_session.date()), "counts": counts}
    )
    ustate_path.parent.mkdir(parents=True, exist_ok=True)
    ustate_path.write_text(json.dumps(ustate, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
