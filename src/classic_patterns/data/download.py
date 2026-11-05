"""Resumable, rate-limit-safe Yahoo daily price download and FRED macro download."""

from __future__ import annotations

import io
import json
import logging
import os
import random
import time
import warnings
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf
from yfinance.exceptions import YFRateLimitError

from classic_patterns.data.adjust import build_price_frame

log = logging.getLogger(__name__)

RATE_LIMIT_MARKERS = (
    "too many requests",
    "rate limit",
    "429",
    "jsondecodeerror",
    "expecting value",
    "currently down",
)
PROBE_SYMBOL = "SPY"

# yfinance triggers pandas-3 deprecation warnings on every call.
warnings.filterwarnings("ignore", module=r"yfinance\.")


def now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class RateLimited(Exception):
    """Yahoo refused the request (rate limit, 429, garbage JSON)."""


class StopDownload(Exception):
    """Too many consecutive failed attempts; state is saved, resume later."""


# ---------------------------------------------------------------------------
# state
# ---------------------------------------------------------------------------


@dataclass
class DownloadState:
    path: Path
    done: dict[str, dict] = field(default_factory=dict)
    empty: dict[str, dict] = field(default_factory=dict)
    failed: dict[str, dict] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> DownloadState:
        if not path.exists():
            return cls(path)
        data = json.loads(path.read_text())
        return cls(path, data.get("done", {}), data.get("empty", {}), data.get("failed", {}))

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        payload = {
            "updated_at": now_iso(),
            "done": self.done,
            "empty": self.empty,
            "failed": self.failed,
        }
        tmp.write_text(json.dumps(payload, indent=1, sort_keys=True))
        os.replace(tmp, self.path)

    def mark_done(self, sym: str, frame: pd.DataFrame) -> None:
        self.failed.pop(sym, None)
        self.empty.pop(sym, None)
        self.done[sym] = {
            "rows": len(frame),
            "first": str(frame["date"].min().date()),
            "last": str(frame["date"].max().date()),
            "at": now_iso(),
        }

    def mark_failed(self, sym: str, error: str) -> None:
        prev = self.failed.get(sym, {})
        self.failed[sym] = {
            "attempts": prev.get("attempts", 0) + 1,
            "error": error,
            "at": now_iso(),
        }

    def mark_empty(self, sym: str, error: str) -> None:
        self.failed.pop(sym, None)
        self.empty[sym] = {"error": error, "at": now_iso()}


# ---------------------------------------------------------------------------
# fetching
# ---------------------------------------------------------------------------


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def fetch(symbols: list[str], yf_cfg: dict) -> dict[str, pd.DataFrame | None]:
    """Download a batch. Returns {symbol: price frame or None if no data}.

    Raises RateLimited if Yahoo refused the request.
    """
    yf_logger = logging.getLogger("yfinance")
    cap = _Capture()
    yf_logger.addHandler(cap)
    old_level, old_prop = yf_logger.level, yf_logger.propagate
    yf_logger.setLevel(logging.ERROR)
    yf_logger.propagate = False
    try:
        raw = yf.download(
            symbols,
            period=yf_cfg["period"],
            interval=yf_cfg["interval"],
            auto_adjust=False,
            actions=True,
            group_by="ticker",
            progress=False,
            threads=yf_cfg["threads"],
        )
    except YFRateLimitError as e:
        raise RateLimited(str(e)) from e
    except (json.JSONDecodeError, requests.RequestException) as e:
        raise RateLimited(repr(e)) from e
    finally:
        yf_logger.removeHandler(cap)
        yf_logger.setLevel(old_level)
        yf_logger.propagate = old_prop

    errors = " ".join(cap.messages).lower()
    if any(m in errors for m in RATE_LIMIT_MARKERS):
        raise RateLimited(" | ".join(cap.messages)[:500])

    out: dict[str, pd.DataFrame | None] = {}
    for sym in symbols:
        sub = None
        if raw is not None and not raw.empty:
            if isinstance(raw.columns, pd.MultiIndex):
                if sym in raw.columns.get_level_values(0):
                    sub = raw[sym]
            elif len(symbols) == 1:
                sub = raw
        frame = build_price_frame(sub, sym) if sub is not None else None
        out[sym] = frame if frame is not None and len(frame) else None
    return out


def probe_ok(yf_cfg: dict) -> bool:
    """Is Yahoo answering at all? Used when a whole batch comes back empty."""
    try:
        res = fetch([PROBE_SYMBOL], {**yf_cfg, "period": "5d"})
    except RateLimited:
        return False
    return res.get(PROBE_SYMBOL) is not None


def out_path(sym: str, dirs: dict[str, Path], benchmarks: set[str]) -> Path:
    d = dirs["benchmarks"] if sym in benchmarks or sym.startswith("^") else dirs["prices"]
    return d / f"{sym}.parquet"


# ---------------------------------------------------------------------------
# main loop
# ---------------------------------------------------------------------------


class Downloader:
    def __init__(self, cfg: dict, state: DownloadState, dirs: dict[str, Path]):
        self.cfg = cfg
        self.state = state
        self.dirs = dirs
        self.benchmarks = set(cfg["benchmarks"])
        self.consecutive_failures = 0
        self.rows_written = 0

    def _save(self, sym: str, frame: pd.DataFrame) -> None:
        path = out_path(sym, self.dirs, self.benchmarks)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        frame.to_parquet(tmp, compression="zstd", index=False)
        os.replace(tmp, path)
        self.state.mark_done(sym, frame)
        self.rows_written += len(frame)

    def _backoff(self, reason: str) -> None:
        self.consecutive_failures += 1
        self.state.save()
        if self.consecutive_failures >= self.cfg["max_consecutive_failures"]:
            raise StopDownload(f"{self.consecutive_failures} consecutive failures; last: {reason}")
        sched = self.cfg["backoff_minutes"]
        minutes = sched[min(self.consecutive_failures - 1, len(sched) - 1)]
        log.warning(
            "attempt failed (%s); backing off %d min (failure %d/%d)",
            reason[:200],
            minutes,
            self.consecutive_failures,
            self.cfg["max_consecutive_failures"],
        )
        time.sleep(minutes * 60)

    def _attempt(self, batch: list[str]) -> list[str]:
        """Download one batch, retrying with backoff. Returns symbols that came back empty."""
        while True:
            try:
                res = fetch(batch, self.cfg["yfinance"])
            except RateLimited as e:
                self._backoff(f"rate limited: {e}")
                continue
            got = [s for s, f in res.items() if f is not None]
            if not got and not probe_ok(self.cfg["yfinance"]):
                self._backoff("whole batch empty and probe failed")
                continue
            self.consecutive_failures = 0
            for s in got:
                self._save(s, res[s])
            missing = [s for s, f in res.items() if f is None]
            for s in missing:
                self.state.mark_failed(s, "no data in batch download")
            self.state.save()
            return missing

    def run(self, symbols: list[str]) -> None:
        todo = [
            s
            for s in dict.fromkeys(symbols)
            if s not in self.state.done and s not in self.state.empty
        ]
        log.info(
            "%d symbols requested, %d already done, %d empty, %d to download",
            len(symbols),
            len(self.state.done),
            len(self.state.empty),
            len(todo),
        )
        bs = self.cfg["batch_size"]
        batches = [todo[i : i + bs] for i in range(0, len(todo), bs)]
        retry: list[str] = []
        t0 = time.monotonic()
        for i, batch in enumerate(batches, 1):
            retry += self._attempt(batch)
            elapsed = time.monotonic() - t0
            eta_min = elapsed / i * (len(batches) - i) / 60
            log.info(
                "batch %d/%d: done=%d failed=%d empty=%d rows=%d ETA %.0f min",
                i,
                len(batches),
                len(self.state.done),
                len(self.state.failed),
                len(self.state.empty),
                self.rows_written,
                eta_min,
            )
            if i < len(batches):
                time.sleep(random.uniform(*self.cfg["sleep_seconds"]))

        if retry:
            log.info("retrying %d symbols individually", len(retry))
        for sym in retry:
            time.sleep(random.uniform(1, 3))
            try:
                res = fetch([sym], self.cfg["yfinance"])
            except RateLimited as e:
                self.state.mark_failed(sym, f"rate limited on retry: {e}")
                self._backoff(f"rate limited: {e}")
                continue
            if res[sym] is not None:
                self._save(sym, res[sym])
            else:
                self.state.mark_empty(sym, "no data on individual retry")
            self.state.save()
        log.info(
            "finished: done=%d failed=%d empty=%d rows written this run=%d",
            len(self.state.done),
            len(self.state.failed),
            len(self.state.empty),
            self.rows_written,
        )


# ---------------------------------------------------------------------------
# FRED
# ---------------------------------------------------------------------------


def download_fred(
    series: str, url_template: str, out_dir: Path, attempts: int = 3, backoff_s: float = 10.0
) -> pd.DataFrame:
    """Download one FRED series (CSV endpoint) and store it as Parquet.

    Sends the default ``requests`` User-Agent: FRED's CSV endpoint stalls requests with a custom
    User-Agent (observed 2026-11). Network errors and HTTP errors are retried ``attempts`` times
    with a doubling pause; the last error is raised."""
    url = url_template.format(series=series)
    for i in range(attempts):
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            break
        except requests.RequestException as e:
            if i == attempts - 1:
                raise
            wait = backoff_s * 2**i
            log.warning(
                "FRED %s: %s (attempt %d/%d), retrying in %.0f s", series, e, i + 1, attempts, wait
            )
            time.sleep(wait)
    df = pd.read_csv(io.StringIO(resp.text), na_values=["."])
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"]).astype("datetime64[ns]")
    df["series"] = series
    df = df.dropna(subset=["value"]).reset_index(drop=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / f"{series}.parquet", compression="zstd", index=False)
    return df
