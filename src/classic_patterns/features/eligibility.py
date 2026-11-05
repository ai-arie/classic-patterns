"""Point-in-time eligibility: a per-day mask, never a deletion of history.

A day is eligible if close_raw >= min_price, 50-day dollar volume >= threshold, the ticker has
>= min_history rows up to and including that day, and the day is not in an exclusion span.
With hysteresis, the price / dollar-volume part switches on as soon as both are met and
switches off only after ``off_days`` consecutive failing days.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def hysteresis(ok: np.ndarray, off_days: int) -> np.ndarray:
    out = np.zeros(len(ok), dtype=bool)
    on = False
    fails = 0
    for i, v in enumerate(ok):
        if v:
            on, fails = True, 0
        elif on:
            fails += 1
            if fails >= off_days:
                on = False
        out[i] = on
    return out


def span_mask(dates, excl: pd.DataFrame) -> np.ndarray:
    """Rows of one ticker inside any ``exclude_span`` of ``excl`` (inclusive bounds)."""
    dates = pd.DatetimeIndex(dates)
    mask = np.zeros(len(dates), dtype=bool)
    for r in excl[excl["action"] == "exclude_span"].itertuples():
        mask |= (dates >= r.start) & (dates <= r.end)
    return mask


def eligibility_parts(close_raw, dollar_vol, in_span, cfg: dict, min_dollar_volume: float):
    """Components and final mask: dict of bool arrays (price, dollar_volume, history, span,
    raw = all conditions without hysteresis, eligible = final)."""
    close_raw = np.asarray(close_raw, dtype=np.float64)
    dollar_vol = np.asarray(dollar_vol, dtype=np.float64)
    price = close_raw >= cfg["min_price"]
    dv = dollar_vol >= min_dollar_volume  # NaN during warm-up -> False
    history = np.arange(len(close_raw)) >= cfg["min_history"] - 1
    ok = price & dv
    gate = hysteresis(ok, cfg["off_days"]) if cfg["hysteresis"] else ok
    span = np.asarray(in_span, dtype=bool)
    return {
        "price": price,
        "dollar_volume": dv,
        "history": history,
        "span": span,
        "raw": ok & history & ~span,
        "eligible": gate & history & ~span,
    }


def dollar_volume_thresholds(cfg: dict) -> list[float]:
    """Default threshold first, then the report variants (bit i of the stored mask = entry i)."""
    return [cfg["min_dollar_volume"], *cfg["dollar_volume_variants"]]
