"""Causal indicator primitives on 1-d float arrays (one row per bar, oldest first).

Every output at row i depends only on rows <= i, and is computed sequentially from the first
row, so a series truncated at row i gives bit-identical values up to row i. Warm-up rows are NaN.
Used for stocks, indices and the app alike.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import lfilter


def _f64(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def sma(x, n: int) -> np.ndarray:
    return pd.Series(_f64(x)).rolling(n, min_periods=n).mean().to_numpy()


def rolling_max(x, n: int) -> np.ndarray:
    return pd.Series(_f64(x)).rolling(n, min_periods=n).max().to_numpy()


def rolling_min(x, n: int) -> np.ndarray:
    return pd.Series(_f64(x)).rolling(n, min_periods=n).min().to_numpy()


def rolling_std(x, n: int) -> np.ndarray:
    return pd.Series(_f64(x)).rolling(n, min_periods=n).std(ddof=1).to_numpy()


def shift(x, k: int) -> np.ndarray:
    """Value k rows earlier (NaN for the first k rows)."""
    x = _f64(x)
    out = np.full_like(x, np.nan)
    if k < len(x):
        out[k:] = x[: len(x) - k]
    return out


def wilder(x, n: int) -> np.ndarray:
    """Wilder smoothing: seed = mean of the first n values, then y = y_prev + (x - y_prev) / n.

    Leading NaNs in ``x`` are skipped; the input must have no NaN after its first valid value.
    """
    x = _f64(x)
    out = np.full_like(x, np.nan)
    valid = np.flatnonzero(~np.isnan(x))
    if len(valid) < n:
        return out
    s = valid[0]
    seed_end = s + n - 1
    out[seed_end] = x[s : seed_end + 1].mean()
    rest = x[seed_end + 1 :]
    if len(rest):
        a = 1.0 / n
        y, _ = lfilter([a], [1.0, -(1.0 - a)], rest, zi=[(1.0 - a) * out[seed_end]])
        out[seed_end + 1 :] = y
    return out


def true_range(high, low, close) -> np.ndarray:
    high, low, close = _f64(high), _f64(low), _f64(close)
    prev = shift(close, 1)
    tr = np.fmax(high - low, np.fmax(np.abs(high - prev), np.abs(low - prev)))
    tr[0] = high[0] - low[0]
    return tr


def atr(high, low, close, n: int = 14) -> np.ndarray:
    return wilder(true_range(high, low, close), n)


def rsi(close, n: int = 14) -> np.ndarray:
    d = np.diff(_f64(close), prepend=np.nan)
    gain = wilder(np.where(np.isnan(d), np.nan, np.maximum(d, 0.0)), n)
    loss = wilder(np.where(np.isnan(d), np.nan, np.maximum(-d, 0.0)), n)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 100.0 - 100.0 / (1.0 + gain / loss)
    out[(loss == 0) & (gain > 0)] = 100.0
    out[(loss == 0) & (gain == 0)] = 50.0
    return out


def rolling_percentile(x, n: int) -> np.ndarray:
    """Share of the last n values (incl. the current one) that are <= the current value."""
    x = _f64(x)
    return (
        pd.Series(x)
        .rolling(n, min_periods=n)
        .apply(lambda w: np.count_nonzero(w <= w[-1]) / len(w), raw=True)
        .to_numpy()
    )


def ratio(a, b) -> np.ndarray:
    """a / b with NaN where b is 0 or NaN."""
    a, b = _f64(a), _f64(b)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = a / b
    out[~np.isfinite(out)] = np.nan
    return out


def pct_dist(x, ref) -> np.ndarray:
    """x / ref - 1."""
    return ratio(x, ref) - 1.0


def trend_block(close, cfg: dict) -> dict[str, np.ndarray]:
    """Trend/momentum indicators shared by stocks and indices (all scale-free)."""
    close = _f64(close)
    out: dict[str, np.ndarray] = {}
    for n in cfg["sma"]:
        out[f"dist_sma{n}_pct"] = pct_dist(close, sma(close, n))
    s_long = sma(close, cfg["slope_sma"])
    name = f"sma{cfg['slope_sma']}_slope{cfg['slope_lag']}"
    out[name] = pct_dist(s_long, shift(s_long, cfg["slope_lag"]))
    out[f"rsi{cfg['rsi']}"] = rsi(close, cfg["rsi"])
    out["drawdown_52w"] = pct_dist(close, rolling_max(close, cfg["hi_lo_window"]))
    return out
