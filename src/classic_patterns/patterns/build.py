"""Pattern events for the whole dataset -> ``artifacts/patterns``.

Outputs:
- ``events.parquet``        default thresholds: one row per (pattern instance, stage)
- ``events_loose.parquet``  labeling thresholds, plus ``failed_default``
- ``meta.json``             config, code hash, data manifest hash, counts

Detection runs on each stock's full continuous history. Per event the table records the
eligibility bitmask on the event date (``elig``, same bits as the feature store), the global
feature-store row (``fs_row``) and ``data_flag`` (the pattern window reaches an exclusion-span
artifact: before the context floor, or a bad-print row).
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

from classic_patterns.data.store import sha256_file
from classic_patterns.patterns.common import CUP_PATTERNS
from classic_patterns.patterns.detect import CONS_PATTERNS, detect_ticker
from classic_patterns.utils.code_hash import code_hash

log = logging.getLogger(__name__)

PRICE_COLS = ["ticker", "date", "open", "high", "low", "close", "volume"]
ROW_COLS = ["date", "elig", "in_span", "floor", "bad", "dollar_vol50"]

_W: dict = {}


def _init(cfg, vix_pct):
    _W.update(cfg=cfg, vix_pct=vix_pct)


def pattern_ids(ev: pd.DataFrame, ticker: str) -> pd.Series:
    d = {c: ev[c].dt.strftime("%Y-%m-%d") for c in ("p1_date", "p3_date", "head_date")}
    cup = ticker + "|" + ev["pattern"] + "|" + d["p1_date"] + "|" + d["p3_date"]
    hs = ticker + "|" + ev["pattern"] + "|" + d["head_date"]
    base = ticker + "|" + ev["pattern"] + "|" + d["p1_date"]  # consolidations: first touch
    out = cup.where(ev["pattern"].isin(CUP_PATTERNS), hs)
    return out.where(~ev["pattern"].isin(CONS_PATTERNS), base)


def _annotate(ev: pd.DataFrame, ticker: str, rows: pd.DataFrame, offset: int) -> pd.DataFrame:
    if ev.empty:
        return ev
    r = ev["row"].to_numpy()
    ev.insert(0, "ticker", ticker)
    ev["pattern_id"] = pattern_ids(ev, ticker)
    ev["event_id"] = ev["pattern_id"] + "|" + ev["stage"]
    ev["fs_row"] = offset + r
    for c in ("elig", "in_span", "dollar_vol50"):
        ev[c] = rows[c].to_numpy()[r]
    nbad = np.concatenate([[0], np.cumsum(rows["bad"].to_numpy())])
    start = ev["start_row"].to_numpy()
    ev["data_flag"] = (start < rows["floor"].to_numpy()[r]) | (nbad[r + 1] - nbad[start] > 0)
    return ev


def _work(args):
    ticker, df, rows, offset = args
    cfg = _W["cfg"]
    assert len(df) == len(rows) and (df["date"].to_numpy() == rows["date"].to_numpy()).all()
    out = []
    for loose in (False, True):
        ev = detect_ticker(df, cfg, _W["vix_pct"], loose=loose)
        out.append(_annotate(ev, ticker, rows, offset))
    return out


def build(cfg: dict, root: Path, workers: int | None = None, tickers: list[str] | None = None):
    t0 = time.perf_counter()
    paths = {k: root / v for k, v in cfg["paths"].items()}
    out_dir = paths["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    workers = workers or cfg["build"]["workers"]

    filters = [("ticker", "in", tickers)] if tickers else None
    prices = pd.read_parquet(paths["prices"], columns=PRICE_COLS, filters=filters)
    fs = paths["features"]
    tick = pd.read_parquet(fs / "tickers.parquet")
    rows = pd.read_parquet(fs / "rows.parquet", columns=["ticker_id", *ROW_COLS])
    vix_pct = pd.read_parquet(fs / "market.parquet").set_index("date")["vix_pct252"]
    offsets = tick.set_index("ticker")["offset"]
    ids = tick.set_index("ticker")["ticker_id"]
    log.info("loaded %d rows, %d tickers", len(prices), prices["ticker"].nunique())

    row_groups = dict(iter(rows.groupby("ticker_id", sort=False)))
    jobs = (
        (t, g.reset_index(drop=True), row_groups[ids[t]].reset_index(drop=True), int(offsets[t]))
        for t, g in prices.groupby("ticker", sort=True)
    )
    n_jobs = prices["ticker"].nunique()
    parts: tuple[list, list] = ([], [])
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(cfg, vix_pct)) as ex:
        for i, res in enumerate(ex.map(_work, jobs, chunksize=4)):
            for p, ev in zip(parts, res, strict=True):
                if not ev.empty:
                    p.append(ev)
            if (i + 1) % 500 == 0:
                log.info("  %d / %d tickers", i + 1, n_jobs)
    counts = {}
    for name, p in zip(("events", "events_loose"), parts, strict=True):
        ev = pd.concat(p, ignore_index=True)
        ev = ev.sort_values(["date", "ticker", "pattern", "pattern_id"], kind="stable")
        ev.reset_index(drop=True).to_parquet(out_dir / f"{name}.parquet", compression="zstd")
        counts[name] = ev.groupby(["pattern", "stage"]).size().unstack().to_dict()
        log.info("%s: %d events", name, len(ev))
    meta = {
        "created_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "code_hash": code_hash(root),
        "data_manifest_sha256": sha256_file(root / "data" / "manifest.json"),
        "feature_build": json.loads((fs / "meta.json").read_text())["created_at"],
        "n_tickers": n_jobs,
        "counts": counts,
        "config": cfg,
        "build_seconds": round(time.perf_counter() - t0, 1),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    return meta
