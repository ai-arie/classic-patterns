"""Causal swing points. Each pivot carries its confirmation row: it may only be used from then on.

- k-bar pivots: bar i is a swing high if its high is above the previous k highs and not below the
  next k highs; it is known at bar i + k.
- zigzag: a swing high is confirmed on the first bar whose low is ``thr`` below the highest high
  since the last swing low (``thr`` = multiple of ATR on that bar), and vice versa. Computed
  sequentially, so a series cut at bar t yields exactly the pivots confirmed by t.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def kbar_peaks(high, k: int) -> np.ndarray:
    """Bool per row: swing high (confirmed at row i + k; False where i + k is past the data)."""
    h = pd.Series(np.asarray(high, dtype=np.float64))
    left = h.shift(1).rolling(k, min_periods=k).max()
    right = h[::-1].shift(1).rolling(k, min_periods=k).max()[::-1]
    return ((h > left) & (h >= right)).to_numpy()


@dataclass
class Zigzag:
    idx: np.ndarray  # pivot row
    confirm: np.ndarray  # row on which the pivot becomes known
    kind: np.ndarray  # +1 swing high, -1 swing low
    price: np.ndarray  # high (kind +1) or low (kind -1) at the pivot


def zigzag(high, low, thr) -> Zigzag:
    """Alternating swing highs/lows with reversal threshold ``thr`` (array, price units)."""
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    thr = np.asarray(thr, dtype=np.float64)
    n = len(high)
    out_i: list[int] = []
    out_c: list[int] = []
    out_k: list[int] = []
    valid = np.flatnonzero(np.isfinite(thr))
    if len(valid) == 0:
        return Zigzag(*(np.zeros(0, dtype=np.int64) for _ in range(3)), np.zeros(0))
    s = int(valid[0])
    mode = 0  # 0 = no pivot yet, +1 = tracking a high, -1 = tracking a low
    hi = lo = ext = s
    for t in range(s + 1, n):
        th = thr[t]
        if not th > 0:
            continue
        if mode == 0:
            if high[t] > high[hi]:
                hi = t
            if low[t] < low[lo]:
                lo = t
            up = lo < t and high[t] - low[lo] >= th
            down = hi < t and high[hi] - low[t] >= th
            if up and (not down or lo < hi):
                out_i.append(lo), out_c.append(t), out_k.append(-1)
                mode, ext = 1, lo + 1 + int(np.argmax(high[lo + 1 : t + 1]))
            elif down:
                out_i.append(hi), out_c.append(t), out_k.append(1)
                mode, ext = -1, hi + 1 + int(np.argmin(low[hi + 1 : t + 1]))
        elif mode == 1:
            if high[t] > high[ext]:
                ext = t
            elif high[ext] - low[t] >= th:
                out_i.append(ext), out_c.append(t), out_k.append(1)
                mode, ext = -1, ext + 1 + int(np.argmin(low[ext + 1 : t + 1]))
        else:
            if low[t] < low[ext]:
                ext = t
            elif high[t] - low[ext] >= th:
                out_i.append(ext), out_c.append(t), out_k.append(-1)
                mode, ext = 1, ext + 1 + int(np.argmax(high[ext + 1 : t + 1]))
    idx = np.array(out_i, dtype=np.int64)
    kind = np.array(out_k, dtype=np.int64)
    price = np.where(kind == 1, high[idx], low[idx]) if len(idx) else np.zeros(0)
    return Zigzag(idx, np.array(out_c, dtype=np.int64), kind, price)


def clean_pivot_bars(b, spike_atr: float | None, gap_atr: float | None) -> np.ndarray:
    """Bool per row: the bar may serve as a pivot. Spike bars (high - low > ``spike_atr`` x ATR14)
    and gap bars (|open - previous close| > ``gap_atr`` x ATR14) are not levels; rows without an
    ATR yet are excluded. ``None`` switches a check off. ``b``: ``patterns.common.Bars``."""
    a = b.atr
    ok = np.isfinite(a) & (a > 0)
    if spike_atr is not None:
        ok &= (b.high - b.low) <= spike_atr * a
    if gap_atr is not None:
        prev = np.concatenate([[np.nan], b.close[:-1]])
        gap = np.abs(b.open - prev)
        ok &= ~(gap > gap_atr * a)  # first row (no previous close) counts as no gap
    return ok
