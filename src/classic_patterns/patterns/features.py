"""Trade-plan features of a detected pattern at the decision close (shared by training and the app).

All in direction-adjusted units of ATR14 at the decision day (same ATR as the detectors):
- ``risk_atr``      distance close -> textbook stop (> 0 = stop on the correct side)
- ``reward_atr``    distance close -> textbook target
- ``rr_planned``    reward / risk (NaN if risk <= 0)
- ``break_ext_atr`` close beyond the trigger level (triggered: how far the break already ran;
                    forming: negative = distance still to go)
- ``risk_pct``      risk as a fraction of the close
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADE_FEATURES = ("risk_atr", "reward_atr", "rr_planned", "break_ext_atr", "risk_pct")


def trade_features(direction, close, atr, stop, target, trigger_price) -> pd.DataFrame:
    d = np.asarray(direction, dtype=np.float64)
    c = np.asarray(close, dtype=np.float64)
    a = np.asarray(atr, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        risk = d * (c - np.asarray(stop, dtype=np.float64)) / a
        reward = d * (np.asarray(target, dtype=np.float64) - c) / a
        out = {
            "risk_atr": risk,
            "reward_atr": reward,
            "rr_planned": np.where(risk > 0, reward / risk, np.nan),
            "break_ext_atr": d * (c - np.asarray(trigger_price, dtype=np.float64)) / a,
            "risk_pct": risk * a / c,
        }
    return pd.DataFrame(out)


def prior_high(high, first_row, row, n: int) -> np.ndarray:
    """Highest high of the ``n`` rows before ``row`` (the decision day itself excluded), not
    before the ticker's ``first_row``; NaN without any prior row. Uses only data up to t-1."""
    h = np.asarray(high, dtype=np.float64)
    out = np.full(len(row), np.nan)
    for i, (f, r) in enumerate(zip(np.asarray(first_row), np.asarray(row), strict=True)):
        a = max(int(f), int(r) - n)
        if a < r:
            out[i] = h[a:r].max()
    return out


def overhead_features(direction, close, atr, target, hi, n: int) -> pd.DataFrame:
    """Overhead resistance of long setups: the prior ``n``-day high ``hi`` above the decision
    close, in ATR14 (``overhead{n}_atr``) and as a fraction of the way to the target
    (``overhead{n}_frac``: < 0 at new highs, 0-1 in the path, >= 1 beyond the target). NaN for
    short setups."""
    d = np.asarray(direction, dtype=np.float64)
    c = np.asarray(close, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        dist = np.where(d > 0, hi - c, np.nan)
        out = {
            f"overhead{n}_atr": dist / np.asarray(atr, dtype=np.float64),
            f"overhead{n}_frac": dist / (np.asarray(target, dtype=np.float64) - c),
        }
    return pd.DataFrame(out)


OVERHEAD_WINDOWS = (252, 126)
OVERHEAD_COLS = tuple(f"overhead{n}_{k}" for n in OVERHEAD_WINDOWS for k in ("atr", "frac"))


def overhead_in_path(oh_atr, oh_frac, clear_atr: float) -> np.ndarray:
    """The prior high lies in the path to the target: more than ``clear_atr`` ATR above the
    close and below the target."""
    a, f = np.asarray(oh_atr, dtype=np.float64), np.asarray(oh_frac, dtype=np.float64)
    return (a > clear_atr) & (f < 1.0)


def event_overhead(high, close, atr, row, direction, target) -> pd.DataFrame:
    """``OVERHEAD_COLS`` for events of one ticker (bars from its first row; ``row`` = decision
    day)."""
    row = np.asarray(row, dtype=np.int64)
    cols = []
    for n in OVERHEAD_WINDOWS:
        hi = prior_high(high, np.zeros(len(row), dtype=np.int64), row, n)
        f = overhead_features(direction, close[row], atr[row], target, hi, n)
        cols.append(f)
    return pd.concat(cols, axis=1)[list(OVERHEAD_COLS)]
