"""Download daily prices (Yahoo), benchmarks and FRED macro series. Resumable.

python scripts/download_prices.py            # full universe + benchmarks + macro
python scripts/download_prices.py --test     # 20-ticker sample from configs/download.yaml
python scripts/download_prices.py --symbols AAPL MSFT
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

import pandas as pd
import yfinance as yf

from classic_patterns.data.download import (
    Downloader,
    DownloadState,
    StopDownload,
    download_fred,
    now_iso,
)
from classic_patterns.utils.code_hash import code_hash
from classic_patterns.utils.config import load_config, resolve
from classic_patterns.utils.logging import setup_logging

log = logging.getLogger("download_prices")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--test", action="store_true", help="download the test tickers only")
    g.add_argument("--symbols", nargs="+", help="download these Yahoo symbols only")
    ap.add_argument("--no-macro", action="store_true", help="skip FRED series")
    ap.add_argument("--config", default="download")
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths = {k: resolve(v) for k, v in cfg["paths"].items()}
    setup_logging(paths["log_file"])
    log.info("yfinance %s, pandas %s, code %s", yf.__version__, pd.__version__, code_hash())

    if args.test:
        symbols = list(cfg["test_tickers"])
    elif args.symbols:
        symbols = args.symbols
    else:
        universe = pd.read_parquet(paths["universe"])
        symbols = list(cfg["benchmarks"]) + universe["yahoo_symbol"].tolist()

    meta_path = paths["state_file"].with_name("download_meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"runs": []}
    meta["runs"].append(
        {
            "started_at": now_iso(),
            "yfinance": yf.__version__,
            "pandas": pd.__version__,
            "code_hash": code_hash(),
            "n_symbols": len(symbols),
            "mode": "test" if args.test else ("symbols" if args.symbols else "full"),
        }
    )
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=1))

    if not args.no_macro:
        for series in cfg["fred"]["series"]:
            df = download_fred(series, cfg["fred"]["url"], paths["raw_macro"])
            log.info(
                "FRED %s: %d obs, %s -> %s",
                series,
                len(df),
                df["date"].min().date(),
                df["date"].max().date(),
            )

    state = DownloadState.load(paths["state_file"])
    dirs = {"prices": paths["raw_prices"], "benchmarks": paths["raw_benchmarks"]}
    try:
        Downloader(cfg, state, dirs).run(symbols)
    except StopDownload as e:
        state.save()
        log.error("stopped cleanly: %s. Re-run the same command later to resume.", e)
        return 2
    except KeyboardInterrupt:
        state.save()
        log.warning("interrupted; state saved. Re-run to resume.")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
