"""Incremental daily update of stored Yahoo prices (screener app; training uses the snapshot).

Per stored ticker a short recent window is fetched and compared with the stored rows it
overlaps. If adjusted or split-adjusted closes differ, or a dividend/split appears in the
window, Yahoo has re-adjusted the history and the ticker is re-downloaded in full.
Otherwise the window replaces the stored tail (new days appended).
"""

from __future__ import annotations

import logging
import os
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from classic_patterns.data import download as dl
from classic_patterns.data.tickers import build_universe

log = logging.getLogger(__name__)

COMPARE_COLS = ("close", "y_close")  # adjusted (dividends + splits) and split-adjusted
EVENT_COLS = ("dividends", "splits")


def merge_recent(
    stored: pd.DataFrame, recent: pd.DataFrame, rtol: float, min_overlap: int
) -> tuple[str, pd.DataFrame | None]:
    """Merge a recent window into the stored history.

    Returns (status, merged) with status ``unchanged`` | ``appended`` | ``refreshed`` |
    ``redownload`` (merged is None for ``redownload``).
    """
    if recent.empty:
        return "unchanged", stored
    s = stored.set_index("date")
    r = recent.set_index("date")
    start = r.index.min()
    overlap = s.index.intersection(r.index)
    if len(overlap) < min_overlap or start < s.index.min():
        return "redownload", None
    if not s.index[s.index >= start].isin(r.index).all():  # stored day missing from window
        return "redownload", None
    for col in COMPARE_COLS:
        a = s.loc[overlap, col].to_numpy(dtype=float)
        b = r.loc[overlap, col].to_numpy(dtype=float)
        if not np.allclose(a, b, rtol=rtol, atol=0.0, equal_nan=True):
            return "redownload", None
    events_stored = s.loc[:, list(EVENT_COLS)].reindex(r.index, fill_value=0.0).to_numpy(float)
    if not np.allclose(events_stored, r.loc[:, list(EVENT_COLS)].to_numpy(float)):
        return "redownload", None

    merged = pd.concat([stored[stored["date"] < start], recent], ignore_index=True)
    if merged.equals(stored):
        return "unchanged", stored
    status = "appended" if merged["date"].max() > stored["date"].max() else "refreshed"
    return status, merged


def truncated_history(
    stored: pd.Series, new: pd.Series, max_start_shift_days: int, min_rows_frac: float
) -> str | None:
    """Why a full re-download (dates ``new``) must not replace the stored history (dates
    ``stored``), or None if it may. After a delisting Yahoo can return only the last few days,
    and a stock's history must never be lost."""
    if len(new) == 0:
        return "is empty"
    shift = (new.min() - stored.min()).days
    if shift > max_start_shift_days:
        return f"starts {new.min().date()}, {shift} days after the stored {stored.min().date()}"
    if len(new) < min_rows_frac * len(stored):
        return f"has {len(new)} rows, stored {len(stored)}"
    return None


class Updater(dl.Downloader):
    """Reuses the downloader's state, saving and rate-limit backoff."""

    def __init__(self, cfg: dict, state: dl.DownloadState, dirs: dict[str, Path], last_session):
        super().__init__(cfg, state, dirs)
        self.last_session = pd.Timestamp(last_session)
        self.ucfg = cfg["update"]
        self.recent_yf = {**cfg["yfinance"], "period": self.ucfg["period"]}
        self.redownloading: set[str] = set()
        self.kept: list[str] = []

    def _save(self, sym: str, frame: pd.DataFrame) -> None:
        # never store a bar of a session that has not closed yet
        frame = frame[frame["date"] <= self.last_session].reset_index(drop=True)
        path = dl.out_path(sym, self.dirs, self.benchmarks)
        if sym in self.redownloading and path.exists():
            why = truncated_history(
                pd.read_parquet(path, columns=["date"])["date"],
                frame["date"],
                self.ucfg.get("redownload_max_start_shift_days", 31),
                self.ucfg.get("redownload_min_rows_frac", 0.9),
            )
            if why:
                log.warning("%s: full re-download %s; stored file kept", sym, why)
                self.kept.append(sym)
                return
        super()._save(sym, frame)

    def _fetch_recent(self, batch: list[str]) -> dict[str, pd.DataFrame | None]:
        while True:
            try:
                res = dl.fetch(batch, self.recent_yf)
            except dl.RateLimited as e:
                self._backoff(f"rate limited: {e}")
                continue
            if all(f is None for f in res.values()) and not dl.probe_ok(self.cfg["yfinance"]):
                self._backoff("whole batch empty and probe failed")
                continue
            self.consecutive_failures = 0
            return res

    def run(self, symbols: list[str]) -> dict[str, int]:
        symbols = list(dict.fromkeys(symbols))
        stored = [s for s in symbols if dl.out_path(s, self.dirs, self.benchmarks).exists()]
        new = [s for s in symbols if s not in set(stored)]
        log.info(
            "update to %s: %d stored symbols, %d without data (full download)",
            self.last_session.date(),
            len(stored),
            len(new),
        )
        if new:
            super().run(new)  # skips symbols already known to be empty on Yahoo

        counts: Counter[str] = Counter()
        redownload: list[str] = []
        bs = self.cfg["batch_size"]
        batches = [stored[i : i + bs] for i in range(0, len(stored), bs)]
        for i, batch in enumerate(batches, 1):
            res = self._fetch_recent(batch)
            for sym in batch:
                recent = res[sym]
                if recent is None:
                    counts["no_data"] += 1
                    log.info("%s: no recent data from Yahoo (delisted?)", sym)
                    continue
                recent = recent[recent["date"] <= self.last_session].reset_index(drop=True)
                path = dl.out_path(sym, self.dirs, self.benchmarks)
                status, merged = merge_recent(
                    pd.read_parquet(path), recent, self.ucfg["rtol"], self.ucfg["min_overlap_days"]
                )
                counts[status] += 1
                if status == "redownload":
                    redownload.append(sym)
                elif status != "unchanged":
                    self._save(sym, merged)
            self.state.save()
            if i % 20 == 0 or i == len(batches):
                log.info("batch %d/%d: %s", i, len(batches), dict(counts))
            if i < len(batches):
                time.sleep(random.uniform(*self.cfg["sleep_seconds"]))

        if redownload:
            log.info("re-downloading %d re-adjusted symbols: %s", len(redownload), redownload)
        self.redownloading = set(redownload)
        for j in range(0, len(redownload), bs):
            missing = self._attempt(redownload[j : j + bs])
            for sym in missing:
                log.warning("%s: full re-download returned no data; stored file kept", sym)
            if j + bs < len(redownload):
                time.sleep(random.uniform(*self.cfg["sleep_seconds"]))
        self.redownloading = set()
        if self.kept:
            counts["redownload_kept"] = len(self.kept)
        counts["new_listings"] = len(new)
        log.info("update finished: %s", dict(counts))
        return dict(counts)


def refresh_universe(universe_cfg: dict, universe_path: Path, changes_path: Path):
    """Rebuild the universe from today's listings and log additions / removals.

    Removed tickers drop out of the universe (so they stop updating) but keep their files.
    """
    old = pd.read_parquet(universe_path)
    new, dropped = build_universe(universe_cfg)
    added = sorted(set(new["yahoo_symbol"]) - set(old["yahoo_symbol"]))
    removed = sorted(set(old["yahoo_symbol"]) - set(new["yahoo_symbol"]))

    for df, path in (
        (new, universe_path),
        (dropped, universe_path.with_name("universe_dropped.parquet")),
    ):
        tmp = path.with_suffix(".tmp")
        df.to_parquet(tmp, compression="zstd", index=False)
        os.replace(tmp, path)

    if added or removed:
        today = pd.Timestamp.now().normalize().as_unit("ns")
        rows = pd.DataFrame(
            {
                "date": today,
                "yahoo_symbol": added + removed,
                "change": ["added"] * len(added) + ["removed"] * len(removed),
            }
        )
        if changes_path.exists():
            rows = pd.concat([pd.read_parquet(changes_path), rows], ignore_index=True)
        rows.to_parquet(changes_path, compression="zstd", index=False)
    log.info(
        "universe refreshed: %d tickers, +%d added, -%d removed", len(new), len(added), len(removed)
    )
    return added, removed
