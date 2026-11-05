"""Feature build for the whole dataset -> training arrays in ``artifacts/features``.

Outputs:
- ``stock_features.npy``  float32 memmap [total_rows, n_channels], tickers concatenated
- ``rows.parquet``        per row: ticker_id, date, elig (bitmask, bit i = dollar-volume
                          threshold i), in_span, floor, bad, ctx_idx, dollar_vol50
- ``tickers.parquet``     ticker_id, ticker, offset, n_rows
- ``market.parquet``      market context per NYSE session (row i = ctx_idx i)
- ``filter_stats.parquet`` stock-days removed by each eligibility filter, per year
- ``meta.json``           channel names, config, code hash, data manifest hash
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from classic_patterns.data.calendar import nyse_sessions
from classic_patterns.data.store import context_guards, sha256_file
from classic_patterns.features.eligibility import (
    dollar_volume_thresholds,
    eligibility_parts,
    span_mask,
)
from classic_patterns.features.market import breadth, market_context
from classic_patterns.features.stock import dollar_volume, stock_features
from classic_patterns.features.weekly import week_end_sessions, weekly_features
from classic_patterns.utils.code_hash import code_hash

log = logging.getLogger(__name__)

PRICE_COLS = ["ticker", "date", "open", "high", "low", "close", "close_raw", "volume", "volume_raw"]
BREADTH_SMAS = (200, 50)

_W: dict = {}  # per-worker globals


def ticker_features(df: pd.DataFrame, bench_close: pd.Series, week_ends, cfg: dict) -> pd.DataFrame:
    """All per-stock channels (daily + weekly) for one ticker: the single feature code path."""
    daily = stock_features(df, bench_close, cfg["stock"])
    weekly = weekly_features(df, week_ends, cfg["weekly"])
    return pd.concat([daily, weekly], axis=1)


def _init(bench_close, week_ends, sessions, cfg):
    _W.update(bench_close=bench_close, week_ends=week_ends, sessions=sessions, cfg=cfg)


def _year_counts(years: np.ndarray, parts: dict, start_year: int) -> dict[str, np.ndarray]:
    sel = years >= start_year
    idx = years[sel] - start_year
    n = 2100 - start_year
    out = {"rows": np.bincount(idx, minlength=n)}
    for k in ("price", "dollar_volume", "history"):
        out[f"fail_{k}"] = np.bincount(idx, weights=(~parts[k][sel]).astype(float), minlength=n)
    for k, name in (("span", "in_span"), ("raw", "eligible_raw"), ("eligible", "eligible")):
        out[name] = np.bincount(idx, weights=parts[k][sel].astype(float), minlength=n)
    return out


def _work(args):
    df, excl = args
    cfg, sessions = _W["cfg"], _W["sessions"]
    ecfg = cfg["eligibility"]
    feats = ticker_features(df, _W["bench_close"], _W["week_ends"], cfg)
    dates = pd.DatetimeIndex(df["date"])
    dv = dollar_volume(df, cfg["stock"]["dollar_volume_window"])
    in_span = span_mask(dates, excl)
    elig = np.zeros(len(df), dtype=np.uint8)
    stats = {}
    start_year = int(cfg["build"]["report_start"][:4])
    years = dates.year.to_numpy()
    for bit, thr in enumerate(dollar_volume_thresholds(ecfg)):
        parts = eligibility_parts(df["close_raw"], dv, in_span, ecfg, thr)
        elig |= parts["eligible"].astype(np.uint8) << bit
        stats[thr] = _year_counts(years, parts, start_year)
    floor, bad = context_guards(dates, excl)
    ctx_idx = sessions.get_indexer(dates)
    assert (ctx_idx >= 0).all(), "row date not an NYSE session"

    # breadth contributions (default eligibility)
    e0 = (elig & 1).astype(bool)
    contrib = {"n_eligible": e0}
    for n in BREADTH_SMAS:
        dist = feats[f"dist_sma{n}_pct"].to_numpy()
        valid = e0 & ~np.isnan(dist)
        contrib[f"n_valid{n}"] = valid
        contrib[f"n_above{n}"] = valid & (dist > 0)
    rows = {
        "elig": elig,
        "in_span": in_span,
        "floor": floor.astype(np.int32),
        "bad": bad,
        "ctx_idx": ctx_idx.astype(np.int32),
        "dollar_vol50": dv.astype(np.float32),
    }
    return feats.to_numpy(dtype=np.float32), list(feats.columns), rows, contrib, stats


def build(cfg: dict, root: Path, workers: int | None = None, tickers: list[str] | None = None):
    t0 = time.perf_counter()
    paths = {k: root / v for k, v in cfg["paths"].items()}
    out_dir = paths["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    workers = workers or cfg["build"]["workers"]

    filters = [("ticker", "in", tickers)] if tickers else None
    prices = pd.read_parquet(paths["prices"], columns=PRICE_COLS, filters=filters)
    bench = pd.read_parquet(paths["benchmarks"])
    macro = pd.read_parquet(paths["macro"])
    excl = pd.read_parquet(paths["exclusions"])
    log.info("loaded %d rows, %d tickers", len(prices), prices["ticker"].nunique())

    first, last = prices["date"].min(), prices["date"].max()
    sessions = nyse_sessions(first, last)
    week_ends = week_end_sessions(first, last)
    rs = cfg["stock"]["rs_benchmark"]
    bench_close = bench[bench["ticker"] == rs].set_index("date")["close"].sort_index()

    groups = [(t, g.reset_index(drop=True)) for t, g in prices.groupby("ticker", sort=True)]
    del prices
    excl_by = {t: g for t, g in excl.groupby("ticker")}
    empty = excl.iloc[0:0]
    sizes = np.array([len(g) for _, g in groups])
    offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    n_total = int(sizes.sum())

    feats_mm = None
    row_parts = {k: [] for k in ("elig", "in_span", "floor", "bad", "ctx_idx", "dollar_vol50")}
    counts = None
    stats_total: dict = {}
    channels = None
    jobs = ((g, excl_by.get(t, empty)) for t, g in groups)
    with ProcessPoolExecutor(
        workers, initializer=_init, initargs=(bench_close, week_ends, sessions, cfg)
    ) as ex:
        for i, (x, cols, rows, contrib, stats) in enumerate(ex.map(_work, jobs, chunksize=8)):
            if feats_mm is None:
                channels = cols
                feats_mm = np.lib.format.open_memmap(
                    out_dir / "stock_features.npy.tmp",
                    mode="w+",
                    dtype=np.float32,
                    shape=(n_total, len(cols)),
                )
                counts = {k: np.zeros(len(sessions), dtype=np.int64) for k in contrib}
            assert cols == channels
            feats_mm[offsets[i] : offsets[i] + len(x)] = x
            for k, v in rows.items():
                row_parts[k].append(v)
            for k, v in contrib.items():
                c = np.bincount(rows["ctx_idx"], weights=v, minlength=len(sessions))
                counts[k] += c.astype(np.int64)
            for thr, s in stats.items():
                acc = stats_total.setdefault(
                    thr, {k: np.zeros_like(v, dtype=np.float64) for k, v in s.items()}
                )
                for k, v in s.items():
                    acc[k] += v
            if (i + 1) % 500 == 0:
                log.info("  %d / %d tickers", i + 1, len(groups))
    feats_mm.flush()
    del feats_mm
    (out_dir / "stock_features.npy.tmp").replace(out_dir / "stock_features.npy")

    tick = pd.DataFrame(
        {
            "ticker_id": np.arange(len(groups), dtype=np.int32),
            "ticker": [t for t, _ in groups],
            "offset": offsets,
            "n_rows": sizes,
        }
    )
    rows = pd.DataFrame(
        {
            "ticker_id": np.repeat(tick["ticker_id"].to_numpy(), sizes),
            "date": np.concatenate([g["date"].to_numpy() for _, g in groups]),
        }
    )
    for k, v in row_parts.items():
        rows[k] = np.concatenate(v)
    rows.to_parquet(out_dir / "rows.parquet", compression="zstd")
    tick.to_parquet(out_dir / "tickers.parquet", compression="zstd")

    market = market_context(bench, macro, sessions, cfg["market"], cfg["stock"]["trend"])
    market = market.join(breadth(pd.DataFrame(counts, index=sessions)))
    market.reset_index().to_parquet(out_dir / "market.parquet", compression="zstd")

    start_year = int(cfg["build"]["report_start"][:4])
    fs = []
    for thr, s in stats_total.items():
        f = pd.DataFrame(s)
        f.insert(0, "year", np.arange(start_year, start_year + len(f)))
        f.insert(1, "min_dollar_volume", thr)
        fs.append(f[f["rows"] > 0])
    filter_stats = pd.concat(fs, ignore_index=True)
    filter_stats.to_parquet(out_dir / "filter_stats.parquet")

    meta = {
        "created_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "code_hash": code_hash(root),
        "data_manifest_sha256": sha256_file(paths["manifest"]),
        "n_rows": n_total,
        "n_tickers": len(groups),
        "channels": channels,
        "market_columns": list(market.columns),
        "elig_bits": dollar_volume_thresholds(cfg["eligibility"]),
        "first_date": str(first.date()),
        "last_date": str(last.date()),
        "config": cfg,
        "build_seconds": round(time.perf_counter() - t0, 1),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    log.info(
        "wrote %s: %d rows x %d channels, %d market columns",
        out_dir,
        n_total,
        len(channels),
        market.shape[1],
    )
    return meta
