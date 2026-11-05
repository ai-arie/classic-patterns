"""Synthetic OHLCV on real NYSE sessions, for fast feature / detector tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from classic_patterns.data.calendar import nyse_sessions


def ohlcv(
    n: int = 1500,
    seed: int = 0,
    start: str = "2015-01-02",
    ticker: str = "SYN",
    drop_rows: int = 5,
    zero_volume_rows: int = 20,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = nyse_sessions(start, pd.Timestamp(start) + pd.Timedelta(days=int(n * 1.6)))[:n]
    r = rng.normal(0.0003, 0.02, n)
    close = 30.0 * np.exp(np.cumsum(r))
    open_ = close * np.exp(rng.normal(0, 0.008, n))
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, 0.01, n)))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 0.01, n)))
    vol = rng.lognormal(13, 0.5, n)
    vol[rng.choice(n, zero_volume_rows, replace=False)] = 0.0
    adj = np.where(np.arange(n) < n // 2, 0.5, 1.0)  # a 2:1 split halfway
    df = pd.DataFrame(
        {
            "ticker": ticker,
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "close_raw": close / adj,
            "volume": vol,
            "volume_raw": vol * adj,
        }
    )
    keep = np.ones(n, dtype=bool)
    keep[rng.choice(np.arange(n // 5, n), drop_rows, replace=False)] = False  # small data gaps
    return df[keep].reset_index(drop=True)


def benchmarks(n: int = 1600, start: str = "2014-06-02", seed: int = 1) -> pd.DataFrame:
    parts = []
    for i, t in enumerate(["^GSPC", "^NDX", "^RUT", "^VIX", "^VIX3M"]):
        df = ohlcv(n, seed + i, start, t, drop_rows=0, zero_volume_rows=0)
        if t == "^VIX3M":
            df = df.iloc[200:]  # starts later, like the real series
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def macro(start: str = "2014-06-02", n_days: int = 2600, seed: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    d = pd.bdate_range(start, periods=n_days)
    return pd.DataFrame(
        {"date": d, "value": np.cumsum(rng.normal(0, 0.02, n_days)), "series": "T10Y2Y"}
    )


def from_path(close, start: str = "2010-01-04", spread: float = 0.3, seed: int = 0) -> pd.DataFrame:
    """Bars around a designed close path: open = previous close, high/low = max/min(open, close)
    +- ``spread``, so swing points sit exactly where the path has its extremes."""
    close = np.asarray(close, dtype=np.float64)
    n = len(close)
    rng = np.random.default_rng(seed)
    dates = nyse_sessions(start, pd.Timestamp(start) + pd.Timedelta(days=int(n * 1.6) + 10))[:n]
    open_ = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame(
        {
            "ticker": "SYN",
            "date": dates,
            "open": open_,
            "high": np.maximum(open_, close) + spread,
            "low": np.minimum(open_, close) - spread,
            "close": close,
            "volume": rng.lognormal(13, 0.2, n),
        }
    )


def _seg(a: float, b: float, n: int) -> np.ndarray:
    """n values moving linearly from a (excluded) to b (included)."""
    return a + (b - a) * np.arange(1, n + 1) / n


def cup_handle_path(depth: float = 0.20) -> tuple[np.ndarray, dict]:
    """Uptrend 50 -> 100 (left lip), U-shaped cup (``depth`` of the left lip) over 120 days to a
    right lip at 98, 10-day handle down to 93, breakout above the right lip."""
    warm = 50 + 0.2 * np.sin(np.arange(300) / 3)
    up = _seg(warm[-1], 100, 150)
    x = np.arange(1, 121) / 120
    low = 100 * (1 - depth)
    cup = 100 - (100 - low) * (1 - (2 * x - 1) ** 2)
    cup[60:] = 98 - (98 - low) * (1 - (2 * x[60:] - 1) ** 2)  # right half rises to 98
    handle = _seg(98, 93, 10)
    brk = _seg(93, 104, 6)
    after = _seg(104, 110, 40)
    path = np.concatenate([warm, up, cup, handle, brk, after])
    L = len(warm) + len(up) - 1
    R = L + 120
    return path, {"L": L, "R": R, "handle_end": R + 10}


def hs_path(top: bool = True) -> tuple[np.ndarray, dict]:
    """Uptrend into LS 110, T1 100, head 125, T2 100, RS 110, then a break below 100
    (mirrored around 100 for the inverse pattern)."""
    warm = 50 + 0.2 * np.sin(np.arange(300) / 3)
    legs = [(100, 100), (110, 20), (100, 20), (125, 20), (100, 20), (110, 20), (90, 20), (85, 30)]
    parts = [warm]
    piv = []
    for level, n in legs:
        parts.append(_seg(parts[-1][-1], level, n))
        piv.append(sum(len(p) for p in parts) - 1)
    path = np.concatenate(parts)
    if not top:
        path = 200.0 - path
    return path, {"LS": piv[1], "T1": piv[2], "H": piv[3], "T2": piv[4], "RS": piv[5]}


def base_path(lows=(94.0, 94.0, 94.0)) -> tuple[np.ndarray, dict]:
    """Uptrend 50 -> 100, then a base under 100: down to each of ``lows`` and back to 100 (10 days
    per leg), then a breakout to 106 and a drift to 110. Touches at the top of each leg."""
    warm = 50 + 0.2 * np.sin(np.arange(300) / 3)
    parts = [warm, _seg(warm[-1], 100, 150)]
    touches = [len(warm) + 150 - 1]
    for lo in lows:
        parts += [_seg(100, lo, 10), _seg(lo, 100, 10)]
        touches.append(sum(len(p) for p in parts) - 1)
    parts += [_seg(100, 106, 6), _seg(106, 110, 40)]
    path = np.concatenate(parts)
    return path, {"touches": touches, "trigger": touches[-1] + 1}


def triangle_path(down: bool = False) -> tuple[np.ndarray, dict]:
    """Uptrend 50 -> 100, then an ascending triangle: flat top at 100, lows 90 / 93 / 96 (legs of
    10, 7 and 4 days), then a break up to 108 or (``down``) a rise to 99 and a fall to 85."""
    warm = 50 + 0.2 * np.sin(np.arange(300) / 3)
    parts = [warm, _seg(warm[-1], 100, 150)]
    lows = []
    for lo, n in ((90, 10), (93, 7), (96, 4)):
        parts.append(_seg(100, lo, n))
        lows.append(sum(len(p) for p in parts) - 1)
        parts.append(_seg(lo, 100, n))
    if down:
        parts += [_seg(100, 98, 3), _seg(98, 85, 10), _seg(85, 80, 30)]
    else:
        parts += [_seg(100, 108, 6), _seg(108, 112, 30)]
    return np.concatenate(parts), {"lows": lows, "first": len(warm) + 150 - 1}
