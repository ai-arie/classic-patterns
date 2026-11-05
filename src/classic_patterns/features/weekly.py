"""Weekly features: weeks end Friday, and a week becomes visible at the close of its last NYSE
session (e.g. Thursday when Friday is a holiday). Until then a day sees only earlier weeks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from classic_patterns.data.calendar import nyse_sessions
from classic_patterns.features import indicators as ind


def week_end_sessions(start, end) -> pd.DatetimeIndex:
    """NYSE sessions in [start, end] that are the last session of their Friday-ending week.

    Uses the exchange calendar beyond ``end`` (holidays are scheduled in advance), not data.
    """
    end = pd.Timestamp(end)
    s = nyse_sessions(start, end + pd.Timedelta(days=14))
    week = s.to_period("W-FRI")
    last = np.append(week[1:] != week[:-1], False)
    return s[last & (s <= end)]


def weekly_features(df: pd.DataFrame, week_ends: pd.DatetimeIndex, cfg: dict) -> pd.DataFrame:
    """Weekly features of the latest completed week, for each daily row of one ticker.

    ``df``: rows sorted by date with date, close, volume. ``week_ends``: from week_end_sessions,
    covering the rows' date range.
    """
    dates = pd.DatetimeIndex(df["date"])
    code = dates.to_period("W-FRI").asi8
    weeks, first, inverse = np.unique(code, return_index=True, return_inverse=True)
    last = np.append(first[1:], len(code)) - 1
    close = df["close"].to_numpy(dtype=np.float64)
    vol = df["volume"].to_numpy(dtype=np.float64)
    w_close = close[last]
    w_vol = np.add.reduceat(vol, first)

    feats: dict[str, np.ndarray] = {f"w_rsi{cfg['rsi']}": ind.rsi(w_close, cfg["rsi"])}
    for n in cfg["sma"]:
        feats[f"w_dist_sma{n}_pct"] = ind.pct_dist(w_close, ind.sma(w_close, n))
    vw = cfg["volume_window"]
    feats[f"w_vol_ratio{vw}"] = ind.ratio(w_vol, ind.shift(ind.sma(w_vol, vw), 1))

    complete = dates.isin(week_ends)
    k = np.where(complete, inverse, inverse - 1)
    out = {}
    for name, v in feats.items():
        col = np.full(len(dates), np.nan)
        ok = k >= 0
        col[ok] = v[k[ok]]
        out[name] = col
    return pd.DataFrame(out, index=df.index)
