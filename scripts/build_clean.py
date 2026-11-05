"""Build data/clean from data/raw and data/meta/exclusions.parquet, then write data/manifest.json.

python scripts/build_clean.py     # run scripts/validate_data.py first

data/clean/prices.parquet      stocks, sorted by ticker, date (exclusion spans are kept;
                               they are masked out of eligibility in the feature build)
data/clean/benchmarks.parquet  indices / ETFs
data/clean/macro.parquet       FRED series (date, value, series)
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from classic_patterns.data.store import ParquetAppender, apply_exclusions, sha256_file
from classic_patterns.utils.code_hash import code_hash, project_root
from classic_patterns.utils.config import load_config, resolve
from classic_patterns.utils.logging import setup_logging

log = logging.getLogger("build_clean")


def build(files, excl: pd.DataFrame, out) -> dict:
    by_ticker = {t: g for t, g in excl.groupby("ticker")}
    empty = excl.iloc[0:0]
    app = ParquetAppender(out)
    tickers, first, last = [], [], []
    for path in sorted(files, key=lambda p: p.stem):
        df = apply_exclusions(pd.read_parquet(path), by_ticker.get(path.stem, empty))
        if df is None:
            continue
        d = df["date"]
        assert d.is_monotonic_increasing and d.is_unique, path.stem
        px = df[["open", "high", "low", "close"]].to_numpy()
        assert np.isfinite(px).all() and (px > 0).all(), path.stem
        app.append(df)
        tickers.append(path.stem)
        first.append(d.iloc[0])
        last.append(d.iloc[-1])
    rows = app.close()
    return {
        "rows": rows,
        "tickers": len(tickers),
        "first_date": str(min(first).date()),
        "last_date": str(max(last).date()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default="validate")
    args = ap.parse_args()
    cfg = load_config(args.config)
    paths = {k: resolve(v) for k, v in cfg["paths"].items()}
    dl_cfg = load_config("download")
    setup_logging(resolve("logs/build_clean.log"))

    clean = resolve("data/clean")
    excl = pd.read_parquet(paths["exclusions"])
    info = {
        "prices.parquet": build(
            paths["raw_prices"].glob("*.parquet"), excl, clean / "prices.parquet"
        ),
        "benchmarks.parquet": build(
            paths["raw_benchmarks"].glob("*.parquet"), excl, clean / "benchmarks.parquet"
        ),
    }
    macro = pd.concat(
        [
            pd.read_parquet(p)
            for p in sorted(resolve(dl_cfg["paths"]["raw_macro"]).glob("*.parquet"))
        ],
        ignore_index=True,
    )
    macro.to_parquet(clean / "macro.parquet", compression="zstd", index=False)
    info["macro.parquet"] = {
        "rows": len(macro),
        "series": sorted(macro["series"].unique().tolist()),
        "first_date": str(macro["date"].min().date()),
        "last_date": str(macro["date"].max().date()),
    }
    for name, i in info.items():
        log.info("%s: %s", name, i)

    root = project_root()
    files = {}
    for p in [*sorted(clean.glob("*.parquet")), *sorted(resolve("data/meta").glob("*.parquet"))]:
        rel = p.relative_to(root).as_posix()
        files[rel] = {
            "bytes": p.stat().st_size,
            "sha256": sha256_file(p),
            "rows": pd.read_parquet(p, columns=[]).shape[0],
            **info.get(p.name, {}),
        }
    state = json.loads(resolve(dl_cfg["paths"]["state_file"]).read_text())
    dl_meta = json.loads(
        resolve(dl_cfg["paths"]["state_file"]).with_name("download_meta.json").read_text()
    )
    manifest = {
        "created_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "code_hash": code_hash(),
        "source": "Yahoo Finance via yfinance (prices), FRED (macro)",
        "download_runs": dl_meta["runs"],
        "download_state_updated_at": state["updated_at"],
        "download_counts": {k: len(state[k]) for k in ("done", "failed", "empty")},
        "exclusions": excl.groupby(["action", "reason"])
        .size()
        .rename("n")
        .reset_index()
        .to_dict("records"),
        "files": files,
    }
    resolve("data/manifest.json").write_text(json.dumps(manifest, indent=1))
    log.info("manifest written: %d files", len(files))


if __name__ == "__main__":
    main()
