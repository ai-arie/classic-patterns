"""Per-stock daily features (adjusted prices unless noted). All features are scale-free, so later
dividend/split re-adjustments of the whole history (a constant factor per date) don't change them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from classic_patterns.features import indicators as ind


def align_benchmark(bench_close: pd.Series, dates) -> np.ndarray:
    """Benchmark close as of each date (last value on or before it)."""
    return bench_close.reindex(pd.DatetimeIndex(dates), method="ffill").to_numpy(dtype=np.float64)


def stock_features(df: pd.DataFrame, bench_close: pd.Series, cfg: dict) -> pd.DataFrame:
    """Daily features for one ticker.

    ``df``: one ticker's rows sorted by date with columns date, open, high, low, close, volume,
    close_raw, volume_raw. ``bench_close``: relative-strength benchmark close indexed by date.
    Returns a frame with the same index as ``df``.
    """
    o, h, lo, c = (df[k].to_numpy(dtype=np.float64) for k in ("open", "high", "low", "close"))
    vol = df["volume"].to_numpy(dtype=np.float64)
    tc = cfg["trend"]
    out: dict[str, np.ndarray] = {}

    out["log_close"] = np.log(c)
    out["ret1"] = np.log(ind.ratio(c, ind.shift(c, 1)))
    atr = ind.atr(h, lo, c, cfg["atr"])
    atr_pct = ind.ratio(atr, c)
    out[f"atr{cfg['atr']}_pct"] = atr_pct

    trend = ind.trend_block(c, tc)
    out[f"rsi{tc['rsi']}"] = trend.pop(f"rsi{tc['rsi']}")
    for n in tc["sma"]:
        out[f"dist_sma{n}_pct"] = trend.pop(f"dist_sma{n}_pct")
    for n in tc["sma"]:
        out[f"dist_sma{n}_atr"] = ind.ratio(c - ind.sma(c, n), atr)
    out.update(trend)  # SMA slope, drawdown_52w
    w = tc["hi_lo_window"]
    out["dist_52w_high"] = ind.pct_dist(c, ind.rolling_max(h, w))
    out["dist_52w_low"] = ind.pct_dist(c, ind.rolling_min(lo, w))

    vw = cfg["volume_window"]
    out[f"vol_ratio{vw}"] = ind.ratio(vol, ind.shift(ind.sma(vol, vw), 1))
    dv = df["close_raw"].to_numpy(dtype=np.float64) * df["volume_raw"].to_numpy(dtype=np.float64)
    out[f"log_dollar_vol{cfg['dollar_volume_window']}"] = np.log1p(
        ind.sma(dv, cfg["dollar_volume_window"])
    )

    for n in cfg["momentum"]:
        out[f"mom{n}"] = ind.pct_dist(c, ind.shift(c, n))
    n12, skip = cfg["momentum_12m"]
    out[f"mom{n12}_skip{skip}"] = ind.pct_dist(ind.shift(c, skip), ind.shift(c, n12))

    rs = ind.ratio(c, align_benchmark(bench_close, df["date"]))
    out["rs_level"] = np.log(ind.ratio(rs, ind.sma(rs, cfg["rs_sma"])))
    out[f"rs_chg{cfg['rs_change']}"] = np.log(ind.ratio(rs, ind.shift(rs, cfg["rs_change"])))

    gap = ind.pct_dist(o, ind.shift(c, 1))
    out["gap"] = gap
    prev_atr_pct = ind.shift(atr_pct, 1)
    big = (np.abs(gap) > cfg["big_gap_atr"] * prev_atr_pct).astype(np.float64)
    big[np.isnan(gap) | np.isnan(prev_atr_pct)] = np.nan
    out["big_gap"] = big
    return pd.DataFrame(out, index=df.index)


def dollar_volume(df: pd.DataFrame, window: int) -> np.ndarray:
    """Mean of close_raw x volume_raw over the last ``window`` rows (incl. today), in dollars."""
    dv = df["close_raw"].to_numpy(dtype=np.float64) * df["volume_raw"].to_numpy(dtype=np.float64)
    return ind.sma(dv, window)
