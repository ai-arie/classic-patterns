"""Cup-and-handle (long). Stages per instance, each emitted on the first day it is reached:

The right lip R is usable once it is a confirmed k-bar swing high (day R + k).

- ``forming``: handle at least ``handle_days[0]`` long and valid, close within ``near_pivot`` of
  the pivot.
- ``triggered``: first close above the pivot (the right lip's high), handle length within
  ``handle_days``. Entry is the next open. The handle low is the lowest low after the right lip.

Left lip L for a right lip R: the nearest earlier k-bar swing high that is the highest high of
[L, R - 1] and gives a lip difference and cup depth within range (cup duration within
``cup_days``). Everything evaluated on day t uses bars <= t only.

Cups deeper than the cup-and-handle limit but at most ``deep_cup_max`` are emitted as the separate
pattern ``deep_cup`` (same gates otherwise); ``deep_cup_max: null`` turns that off.
"""

from __future__ import annotations

import numpy as np

from classic_patterns.patterns.common import (
    Bars,
    at_least,
    at_most,
    in_range,
    lin_slope,
    quad_r2,
    volume_ratio,
)
from classic_patterns.patterns.pivots import clean_pivot_bars, kbar_peaks

PATTERN = "cup_handle"
DEEP = "deep_cup"


def depth_max(c: dict, vix_low: float) -> float:
    """Cup-and-handle depth limit (optionally larger in high-VIX markets)."""
    hv = c.get("cup_depth_max_high_vix")
    if hv is not None and np.isfinite(vix_low) and vix_low >= c["high_vix_pct"]:
        return hv
    return c["cup_depth"][1]


def search_depth_max(c: dict, vix_low: float) -> float:
    """Deepest cup the left-lip search accepts (cup-and-handle or deep cup)."""
    return max(depth_max(c, vix_low), c.get("deep_cup_max") or 0.0)


def is_deep(g: dict, c: dict) -> bool:
    return bool(g["cup_depth"] > depth_max(c, g["cup_low_vix_pct"]))


def low_vix(b: Bars, L: int, R: int, cl: int, c: dict) -> float:
    """Highest VIX 1-year percentile within ``high_vix_window`` days of the cup low, inside the
    cup."""
    w = c["high_vix_window"]
    vp = b.vix_pct[max(L, cl - w) : min(R, cl + w) + 1]
    return float(np.nanmax(vp)) if np.isfinite(vp).any() else np.nan


def gate_failures(g: dict, c: dict, stage: str, deep: bool = False) -> list[str]:
    """Names of the gates that geometry ``g`` fails under config ``c`` at ``stage``, as a
    cup-and-handle or (``deep``) as a deep cup."""
    f = []
    if not at_least(g["prior_rise"], c["prior_rise_min"]):
        f.append("prior_rise")
    if c["lip_high_tol"] is not None and not at_least(g["lip_vs_high"], -c["lip_high_tol"]):
        f.append("lip_vs_high")
    if not in_range(g["cup_days"], c["cup_days"]):
        f.append("cup_days")
    dmax = depth_max(c, g["cup_low_vix_pct"])
    if deep:
        dm = c.get("deep_cup_max")
        ok = dm is not None and dmax < g["cup_depth"] <= dm
    else:
        ok = in_range(g["cup_depth"], (c["cup_depth"][0], dmax))
    if not ok:
        f.append("cup_depth")
    if not in_range(g["lip_diff"], c["lip_diff"]):
        f.append("lip_diff")
    if not at_least(g["bottom_frac"], c["bottom_frac_min"]):
        f.append("bottom_frac")
    if not at_least(g["lip_recovery"], c.get("lip_recovery_min")):
        f.append("lip_recovery")
    if not at_least(g["cup_low_vs_trend"], c.get("cup_low_vs_trend_min")):
        f.append("cup_low_vs_trend")
    if not at_most(g["spike_lips"], c.get("spike_lips_max")):
        f.append("spike_lips")
    if not at_most(g["handle_depth"], c["handle_depth_max"]):
        f.append("handle_depth")
    if not at_most(g["handle_pos"], c["handle_pos_max"]):
        f.append("handle_pos")
    hmin, hmax = c["handle_days"]
    hd = g["handle_days"]
    if hd > hmax or (stage != "alive" and hd < hmin):
        f.append("handle_days")
    if stage != "alive" and not at_most(g["handle_slope"], c["handle_slope_max"]):
        f.append("handle_slope")
    return f


def _left_lip(b: Bars, peaks: np.ndarray, R: int, c: dict) -> int | None:
    lo_d, hi_d = c["cup_days"]
    lo, hi = max(0, R - hi_d), R - lo_d
    if hi < lo:
        return None
    seg = b.high[lo:R]
    # running max from the right: L is a highest high of [L, R-1] (ties: all tied bars)
    after = np.maximum.accumulate(seg[::-1])[::-1]
    after = np.append(after[1:], -np.inf)
    cand = np.flatnonzero(seg >= after)[::-1] + lo  # nearest first
    hr = b.high[R]
    for L in cand:
        if L > hi or not peaks[L]:
            continue
        lip = hr / b.high[L] - 1.0
        if lip > c["lip_diff"][1]:
            continue  # farther candidates are higher
        if lip < c["lip_diff"][0]:
            return None
        cl = L + 1 + int(np.argmin(b.low[L + 1 : R]))
        depth = 1.0 - b.low[cl] / b.high[L]
        if depth > search_depth_max(c, low_vix(b, L, R, cl, c)):
            return None
        if depth < c["cup_depth"][0]:
            continue
        if b.high[cl:R].max(initial=-np.inf) > hr:
            continue  # R must top the right side of the cup
        return int(L)
    return None


def _cup_geometry(b: Bars, L: int, R: int, c: dict, clean: np.ndarray) -> dict:
    cl = L + 1 + int(np.argmin(b.low[L + 1 : R]))
    hl, hr, low = b.high[L], b.high[R], b.low[cl]
    w = b.low[max(0, L - c["prior_window"]) : L]
    wh = b.high[max(0, L - c["lip_high_window"]) : L]
    rim = min(hl, hr)
    closes = b.close[L : R + 1]
    vp = b.vix_pct[L : R + 1]
    return {
        "cup_low_idx": cl,
        "prior_rise": hl / w.min() - 1.0 if len(w) else np.nan,
        "lip_vs_high": hl / wh.max() - 1.0 if len(wh) else np.nan,
        "cup_days": R - L,
        "cup_depth": 1.0 - low / hl,
        "lip_diff": hr / hl - 1.0,
        "lip_recovery": (hr - low) / (hl - low),  # share of the cup depth regained at R
        "cup_low_vs_trend": low / w.min() - 1.0 if len(w) else np.nan,  # vs prior-uptrend start
        "spike_lips": float((~clean[L]) + (~clean[R])),  # lips on spike / gap bars
        "bottom_frac": float(np.mean(closes <= low + (rim - low) / 3.0)),
        "cup_r2": quad_r2(closes),
        "cup_low_pos": (cl - L) / (R - L),
        "cup_vix_pct_max": float(np.nanmax(vp)) if np.isfinite(vp).any() else np.nan,
        "cup_low_vix_pct": low_vix(b, L, R, cl, c),
        "cup_vol_ratio": volume_ratio(b.volume, L, R, L, c["volume_base"]),
    }


def detect(b: Bars, cfg: dict) -> list[dict]:
    """All cup-and-handle stage events of one ticker (list of dicts, unsorted)."""
    c = cfg["cup_handle"]
    k = cfg["pivots"]["k"]
    n = len(b)
    peaks = kbar_peaks(b.high, k)
    clean = clean_pivot_bars(b, c.get("spike_atr"), c.get("gap_atr"))
    hmin, hmax = c["handle_days"]
    events: list[dict] = []
    for R in np.flatnonzero(peaks):
        R = int(R)
        cR = R + k
        if cR >= n:
            break
        L = _left_lip(b, peaks, R, c)
        if L is None:
            continue
        cup = _cup_geometry(b, L, R, c, clean)
        deep = is_deep(cup, c)
        pat = DEEP if deep else PATTERN
        emitted: set[str] = set()
        prev = None  # handle state as of t-1
        for t in range(cR, min(n, R + hmax + 1)):
            if prev is not None and b.close[t] > prev["pivot"] and t - R >= hmin:
                g = {**cup, **prev, "handle_days": t - R}
                if not gate_failures(g, c, "triggered", deep):
                    events.append(_event(b, L, R, t, "triggered", g, c, pat))
                break
            h = _handle(b, R, t, cup, c)
            g = {**cup, **h}
            if gate_failures(g, c, "alive", deep):
                break  # handle too deep / too low / too long: pattern dead
            if (
                "forming" not in emitted
                and b.close[t] >= h["pivot"] * (1.0 - c["near_pivot"])
                and not gate_failures(g, c, "forming", deep)
            ):
                events.append(_event(b, L, R, t, "forming", g, c, pat))
                emitted.add("forming")
            prev = h
    return events


def _handle(b: Bars, R: int, t: int, cup: dict, c: dict) -> dict:
    """Handle state with bars R..t: pivot = the right lip's high, handle low = lowest low after
    the right lip (t > R always: R is usable from R + k)."""
    hr = b.high[R]
    cup_low = b.low[cup["cup_low_idx"]]
    pivot = hr
    j = R + 1 + int(np.argmin(b.low[R + 1 : t + 1]))
    hlow = b.low[j]
    return {
        "pivot": pivot,
        "handle_low": hlow,
        "handle_low_idx": j,
        "handle_days": t - R,
        "handle_depth": 1.0 - hlow / pivot,
        "handle_pos": (hr - hlow) / (hr - cup_low),
        "handle_slope": lin_slope(b.close[R : t + 1]),
        "handle_vol_ratio": volume_ratio(b.volume, R + 1, t, R, c["volume_base"]),
    }


def _event(b: Bars, L: int, R: int, t: int, stage: str, g: dict, c: dict, pattern: str) -> dict:
    depth_px = b.high[L] - b.low[g["cup_low_idx"]]
    return {
        "pattern": pattern,
        "stage": stage,
        "row": t,
        "start_row": L,
        "head_row": -1,
        "p1_row": L,
        "p2_row": g["cup_low_idx"],
        "p3_row": R,
        "p4_row": int(g["handle_low_idx"]),
        "p5_row": -1,
        "trigger_price": g["pivot"],
        "trigger_slope": 0.0,
        "stop": g["handle_low"],
        "target": g["pivot"] + depth_px,
        "vol_ratio_t": volume_ratio(b.volume, t, t, t, c["volume_base"]),
        "pid": f"{pattern}:{L}:{R}",
        **{k: v for k, v in g.items() if k not in ("cup_low_idx", "handle_low_idx", "pivot")},
        "handle_cup_ratio": g["handle_days"] / g["cup_days"],
    }
