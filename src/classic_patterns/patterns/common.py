"""Shared pieces of the pattern detectors: input arrays, stage names, gate checks, config merge."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd

from classic_patterns.features import indicators as ind

STAGES = ("forming", "triggered")
# patterns of the hand-labeling sample
PATTERNS = ("cup_handle", "hs_top", "inv_hs", "consolidation_breakout", "ascending_triangle")
# deep_cup: a cup-and-handle candidate deeper than the cup-and-handle limit (a correction that
# recovers to the old high); detected and evaluated separately, never labeled as cup-and-handle
ALL_PATTERNS = (*PATTERNS, "deep_cup")
CUP_PATTERNS = ("cup_handle", "deep_cup")
# trade direction, +1 long / -1 short (ascending_triangle: per event, by breakout side)
DIRECTION = {
    "cup_handle": 1,
    "hs_top": -1,
    "inv_hs": 1,
    "consolidation_breakout": 1,
    "ascending_triangle": 1,
    "deep_cup": 1,
}


@dataclass
class Bars:
    """One ticker's daily bars as float arrays (adjusted prices, split-adjusted volume)."""

    dates: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    atr: np.ndarray
    vix_pct: np.ndarray  # VIX 1-year percentile per row (NaN if unknown)

    @classmethod
    def from_frame(cls, df: pd.DataFrame, atr_n: int = 14, vix_pct: pd.Series | None = None):
        f = {k: df[k].to_numpy(dtype=np.float64) for k in ("open", "high", "low", "close")}
        dates = pd.DatetimeIndex(df["date"])
        if vix_pct is None:
            vp = np.full(len(df), np.nan)
        else:
            vp = vix_pct.reindex(dates).to_numpy(dtype=np.float64)
        return cls(
            dates=dates.to_numpy(),
            volume=df["volume"].to_numpy(dtype=np.float64),
            atr=ind.atr(f["high"], f["low"], f["close"], atr_n),
            vix_pct=vp,
            **f,
        )

    def __len__(self) -> int:
        return len(self.close)


def merged(cfg: dict, loose: bool) -> dict:
    """Detector config, with the ``labeling`` overrides merged in when ``loose``."""
    out = copy.deepcopy(cfg)
    if loose:
        for sec, over in cfg.get("labeling", {}).items():
            out[sec].update(over)
    return out


def in_range(x: float, rng) -> bool:
    if rng is None:
        return True
    lo, hi = rng
    return bool(np.isfinite(x)) and (lo is None or x >= lo) and (hi is None or x <= hi)


def at_least(x: float, lo) -> bool:
    return lo is None or (bool(np.isfinite(x)) and x >= lo)


def at_most(x: float, hi) -> bool:
    return hi is None or (bool(np.isfinite(x)) and x <= hi)


def mean_or_nan(x: np.ndarray) -> float:
    return float(x.mean()) if len(x) else np.nan


def volume_ratio(vol: np.ndarray, a: int, b: int, base_end: int, base_n: int) -> float:
    """Mean volume over rows [a, b] / mean volume over the ``base_n`` rows before ``base_end``."""
    base = vol[max(0, base_end - base_n) : base_end]
    m = mean_or_nan(base)
    num = mean_or_nan(vol[max(0, a) : b + 1])
    return num / m if m > 0 else np.nan


def lin_slope(y: np.ndarray) -> float:
    """Least-squares slope of y against 0..n-1, divided by mean(y) (fraction per row)."""
    n = len(y)
    if n < 2:
        return np.nan
    x = np.arange(n, dtype=np.float64)
    x -= x.mean()
    return float((x * (y - y.mean())).sum() / (x * x).sum() / y.mean())


def quad_r2(y: np.ndarray) -> float:
    """R^2 of a quadratic fit of y against time."""
    n = len(y)
    if n < 4 or np.ptp(y) == 0:
        return np.nan
    x = np.linspace(-1.0, 1.0, n)
    coef = np.polyfit(x, y, 2)
    res = y - np.polyval(coef, x)
    return float(1.0 - (res * res).sum() / ((y - y.mean()) ** 2).sum())
