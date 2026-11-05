"""Consolidation breakout at highs (long) and ascending triangle (breaks either way).

Measures, stages and reason codes are set in ``configs/detector.yaml`` and below.

Resistance: swing highs are ``k``-bar pivots (usable from bar i + k), at least ``min_spacing``
bars apart. Spike bars (range > ``spike_atr`` ATR) and gap bars (open more than ``gap_atr`` ATR
from the previous close) are never pivots. A touch's level is its high capped at the body top +
``wick_cap_atr`` ATR, so a single long wick does not set the line. For each swing high j (the
latest touch when the instance starts, at j + k), earlier swing highs within ``touch_tol_atr`` ATR
of its level are collected back to the first swing high that is clearly higher (overhead supply
ends the base); the resistance is the highest touch level.

From then on, day by day (bars <= t only): swing highs confirmed on day t within the tolerance
become touches (one clearly higher ends the instance), swing lows confirmed on day t after the
first touch are added; the instance ends on a trigger, when price falls deeper than
``depth_cap`` below the resistance, or ``max_days`` after the first touch.

- consolidation_breakout: triggered = first close above the resistance; forming = close within
  ``near_atr`` ATR below it (once per instance).
- ascending_triangle: support = the line from the first swing low after the first touch with the
  smallest slope to a later swing low (no swing low below it). Triggered = close above the
  resistance (``breakout_dir`` +1) or below the support (-1); forming = >= 2 lows and close within
  ``near_atr`` ATR below the resistance.

Instances are keyed by the first touch; per key only the first event of each stage is kept and a
stage only after the earlier ones (as for head-and-shoulders), so the output is
truncation-invariant.
"""

from __future__ import annotations

import numpy as np

from classic_patterns.patterns.common import Bars, at_least, at_most, in_range, volume_ratio
from classic_patterns.patterns.features import overhead_features, overhead_in_path, prior_high
from classic_patterns.patterns.pivots import clean_pivot_bars, kbar_peaks

CB = "consolidation_breakout"
AT = "ascending_triangle"
SUBTYPES = ("flat_base", "ascending", "tightening", "flag")


def touch_levels(b: Bars, c: dict) -> np.ndarray:
    body = np.maximum(b.open, b.close)
    return np.minimum(b.high, body + c["wick_cap_atr"] * b.atr)


def gate_failures(g: dict, c: dict, stage: str, pattern: str) -> list[str]:
    """Gates of ``pattern`` (its config section ``c``) that geometry ``g`` fails at ``stage``."""
    f = []
    if not at_least(g["touches"], c["touches_min"]):
        f.append("touches")
    if not at_most(g["flatness_atr"], c["flatness_max_atr"]):
        f.append("flatness")
    if not in_range(g["duration_days"], c["duration"]):
        f.append("duration")
    if not at_least(g["height_atr"], c["height_min_atr"]):
        f.append("height")
    if pattern == CB:
        if not at_least(g["prior_rise"], c["prior_rise_min"]):
            f.append("prior_rise")
        if not at_least(g["res_vs_high"], c["res_vs_high_min"]):
            f.append("res_vs_high")
        if not at_most(g["base_depth"], c["depth_max"]):
            f.append("depth")
        if not at_most(g["height_atr"], c.get("height_max_atr")):
            f.append("height_max")
        lt = c["lows_trend_min_atr"]
        if lt is not None and g["n_lows"] >= 2 and not g["lows_trend_atr"] >= lt:
            f.append("lows_trend")
        if not at_most(g["closes_out"], c["closes_out_max"]):
            f.append("containment")
    else:
        if not at_least(g["n_lows"], c["lows_min"]):
            f.append("lows")
        if not at_least(g["lows_rise_min_atr"], c["lows_rise_min_atr"]):
            f.append("lows_rising")
        if stage == "triggered" and not at_most(g["breakout_pos"], c["breakout_pos_max"]):
            f.append("breakout_pos")
        og = c.get("overhead_gate")  # upside breaks only (NaN for downside breaks)
        if og and stage == "triggered" and f"overhead{og['window']}_atr" in g:
            n = og["window"]
            if overhead_in_path(g[f"overhead{n}_atr"], g[f"overhead{n}_frac"], og["clear_atr"]):
                f.append("overhead")
        if not at_most(g["closes_out"], c["closes_out_max"]):
            f.append("containment")
    return f


class _Ctx:
    """Per-ticker arrays shared by all instances."""

    def __init__(self, b: Bars, cfg: dict):
        c = cfg["consolidation"]
        self.b, self.c = b, c
        self.k = c["k"]
        ok = clean_pivot_bars(b, c["spike_atr"], c["gap_atr"])
        self.peaks = np.flatnonzero(kbar_peaks(b.high, self.k) & ok)
        self.troughs = np.flatnonzero(kbar_peaks(-b.low, self.k) & ok)
        self.lvl = touch_levels(b, c)


def _start(x: _Ctx, jj: int):
    """Touches (rows, ascending) for swing high number ``jj`` as the latest touch, or None."""
    b, c, lvl = x.b, x.c, x.lvl
    j = int(x.peaks[jj])
    cj = j + x.k
    tol = c["touch_tol_atr"] * b.atr[cj]
    if not np.isfinite(tol):
        return None
    touches = [j]
    for i in x.peaks[:jj][::-1]:
        i = int(i)
        if j - i > c["max_days"]:
            break
        if lvl[i] > lvl[j] + tol:
            break  # overhead high: the base starts after it
        if abs(lvl[i] - lvl[j]) <= tol:
            if touches[-1] - i >= c["min_spacing"]:
                touches.append(i)
            elif len(touches) > 1 and lvl[i] > lvl[touches[-1]]:
                touches[-1] = i  # two touches too close: keep the higher one (never j itself)
    if len(touches) < 2:
        return None
    return touches[::-1]


def _geometry(x: _Ctx, touches, lows, t: int, R: float, pattern: str) -> dict:
    b, c = x.b, x.c
    first = touches[0]
    a = b.atr[t]
    seg_lo = b.low[first:t]
    base_low_idx = first + int(np.argmin(seg_lo))
    base_low = b.low[base_low_idx]
    lv = x.lvl[touches]
    pw = b.low[max(0, first - c["prior_window"]) : first]
    ph = b.high[max(0, first - c["prior_window"]) : first]
    hw = b.high[max(0, first - c["high_window"]) : first]
    td = c["tight_days"]
    rec = slice(max(first, t - td), t)
    lo_px = np.array([b.low[q] for q in lows]) if lows else np.zeros(0)
    rises = np.diff(lo_px) / a if len(lo_px) >= 2 else np.zeros(0)
    g = {
        "touches": len(touches),
        "flatness_atr": (lv.max() - lv.min()) / a,
        "res_vs_high": R / hw.max() - 1.0 if len(hw) else np.nan,
        "duration_days": t - first,
        "base_depth": 1.0 - base_low / R,
        "height_atr": (R - base_low) / a,
        "n_lows": len(lows),
        "lows_trend_atr": (lo_px[-1] - lo_px[0]) / a if len(lo_px) >= 2 else np.nan,
        "lows_rise_min_atr": rises.min() if len(rises) else np.nan,
        "tightness": (b.high[rec].max() - b.low[rec].min()) / (R - base_low)
        if t > first and R > base_low
        else np.nan,
        "base_vol_ratio": volume_ratio(b.volume, first, t - 1, first, c["volume_base"]),
        "prior_rise": R / pw.min() - 1.0 if len(pw) else np.nan,
        "prior_drop": R / ph.max() - 1.0 if len(ph) else np.nan,
        "base_low_idx": base_low_idx,
        "base_low": base_low,
        "breakout_pos": np.nan,
        "support_slope_atr": np.nan,
    }
    if pattern == AT:
        g["height_atr"] = (R - lo_px[0]) / a if len(lo_px) else np.nan
        g["base_depth"] = 1.0 - lo_px[0] / R if len(lo_px) else np.nan
        if len(lo_px) >= 2:
            s1 = lows[0]
            m, _ = support_line(b, lows)
            g["support_slope_atr"] = m / a
            g["support_m"] = m
            if m > 0:
                apex = s1 + (R - lo_px[0]) / m
                g["breakout_pos"] = (t - first) / (apex - first) if apex > first else np.nan
    sub = "flat_base"
    if g["duration_days"] < 15 and len(lo_px) >= 2 and g["lows_trend_atr"] < 0:
        sub = "flag"
    elif len(rises) and rises.min() >= 0.5:
        sub = "ascending"
    elif g["tightness"] <= 0.5:
        sub = "tightening"
    g["subtype"] = sub
    g["subtype_code"] = float(SUBTYPES.index(sub))
    return g


def support_line(b: Bars, lows) -> tuple[float, int]:
    """Support of the triangle: the line from the first swing low with the smallest slope to any
    later swing low, so no swing low lies below it. Returns (slope per row, the low it rests on)."""
    s1 = lows[0]
    slopes = [(b.low[q] - b.low[s1]) / (q - s1) for q in lows[1:]]
    i = int(np.argmin(slopes))
    return float(slopes[i]), int(lows[1 + i])


def _support(b: Bars, lows, t: int) -> float:
    m, _ = support_line(b, lows)
    return b.low[lows[0]] + m * (t - lows[0])


def _closes_out(x: _Ctx, touches, lows, t: int, R: float, pattern: str) -> int:
    """Closes outside the range before day t: above the resistance (from the first touch), and
    for the triangle also below the support (from the first swing low)."""
    b = x.b
    first = touches[0]
    n = int((b.close[first:t] > R).sum())
    if pattern == AT and len(lows) >= 2:
        s1 = lows[0]
        m, _ = support_line(b, lows)
        sup = b.low[s1] + m * (np.arange(s1, t) - s1)
        n += int((b.close[s1:t] < sup).sum())
    return n


def _scan(x: _Ctx, jj: int, pattern: str, cfg: dict) -> list[dict]:
    b, c = x.b, x.c
    pc = cfg[pattern]
    touches = _start(x, jj)
    if touches is None:
        return []
    n = len(b)
    k = x.k
    first = touches[0]
    j = touches[-1]
    R = float(x.lvl[touches].max())
    tol = c["touch_tol_atr"] * b.atr[j + k]
    lows: list[int] = []
    pk = int(np.searchsorted(x.peaks, j, side="right"))
    tq = int(np.searchsorted(x.troughs, first, side="right"))
    last = min(n - 1, first + c["max_days"])
    out: list[dict] = []
    formed = False
    tries = 0
    for t in range(j + k, last + 1):
        # pivots confirmed on day t
        while pk < len(x.peaks) and x.peaks[pk] + k <= t:
            p = int(x.peaks[pk])
            pk += 1
            if x.lvl[p] > R + tol:
                return out  # a clearly higher swing high: no longer this base
            if abs(x.lvl[p] - R) <= tol:
                if p - touches[-1] >= c["min_spacing"]:
                    touches.append(p)
                elif x.lvl[p] > x.lvl[touches[-1]]:
                    touches[-1] = p  # too close to the last touch: keep the higher one
                R = max(R, float(x.lvl[p]))
        while tq < len(x.troughs) and x.troughs[tq] + k <= t:
            q = int(x.troughs[tq])
            tq += 1
            if q > first:
                lows.append(q)
        if b.low[t] < R * (1.0 - c["depth_cap"]):
            return out
        a = b.atr[t]
        if not np.isfinite(a) or a <= 0:
            continue
        up = b.close[t] > R
        down = pattern == AT and len(lows) >= 2 and b.close[t] < _support(b, lows, t)
        if up or down:
            g = _geometry(x, touches, lows, t, R, pattern)
            g["closes_out"] = _closes_out(x, touches, lows, t, R, pattern)
            if pattern == AT and up:
                g.update(_overhead(b, lows, t, R, g, (pc.get("overhead_gate") or {}).get("window")))
            if not gate_failures(g, pc, "triggered", pattern):
                out.append(_event(b, pattern, "triggered", t, touches, lows, R, g, 1 if up else -1))
            return out
        near = R - b.close[t] <= c["near_atr"] * a
        if formed or not near or t - first < pc["duration"][0]:
            continue
        if pattern == AT and len(lows) < 2:
            continue
        if tries >= c["forming_tries"]:
            continue
        tries += 1
        g = _geometry(x, touches, lows, t, R, pattern)
        g["closes_out"] = _closes_out(x, touches, lows, t, R, pattern)
        if not gate_failures(g, pc, "forming", pattern):
            out.append(_event(b, pattern, "forming", t, touches, lows, R, g, 1))
            formed = True
    return out


def _overhead(b: Bars, lows, t: int, R: float, g: dict, n: int | None) -> dict:
    """Overhead-gate inputs of an upside triangle break at ``t`` (same target as ``_event``, same
    values as the event's ``overhead{n}_*`` columns); empty without a configured window."""
    if not n:
        return {}
    low1 = b.low[lows[0]] if lows else g["base_low"]
    hi = prior_high(b.high, np.array([0]), np.array([t]), n)
    f = overhead_features([1.0], b.close[[t]], b.atr[[t]], [R + (R - low1)], hi, n)
    return {k: float(f[k].iloc[0]) for k in f.columns}


def _event(b: Bars, pattern: str, stage: str, t: int, touches, lows, R, g: dict, d: int) -> dict:
    first = touches[0]
    if pattern == CB:
        stop = b.low[lows[-1]] if lows else g["base_low"]
        height = R - g["base_low"]
        trigger, slope, target = R, 0.0, R + height
        p2, p4 = g["base_low_idx"], (lows[-1] if lows else -1)
    else:
        low1 = b.low[lows[0]] if lows else g["base_low"]
        height = R - low1
        # p4: the swing low the support line rests on (drawn through p2 and p4)
        p2, p4 = (lows[0] if lows else -1), (support_line(b, lows)[1] if len(lows) >= 2 else -1)
        if d > 0:
            trigger, slope = R, 0.0
            stop, target = (b.low[lows[-1]] if lows else g["base_low"]), R + height
        else:
            trigger, slope = _support(b, lows, t), g["support_m"]
            stop, target = R, trigger - height
    gap = (b.open[t] - trigger) * d / b.atr[t] if stage == "triggered" else np.nan
    keep = {k: v for k, v in g.items() if k not in ("base_low_idx", "base_low", "support_m")}
    return {
        "pattern": pattern,
        "stage": stage,
        "row": t,
        "start_row": first,
        "head_row": -1,
        "p1_row": first,
        "p2_row": int(p2),
        "p3_row": int(touches[-1]),
        "p4_row": int(p4),
        "p5_row": -1,
        "trigger_price": trigger,
        "trigger_slope": slope,
        "stop": stop,
        "target": target,
        "pid": f"{pattern}:{first}",
        "trade_dir": float(d),
        "breakout_dir": float(d) if stage == "triggered" else np.nan,
        "gap_atr_t": max(gap, 0.0) if np.isfinite(gap) else np.nan,
        "resistance": R,
        **keep,
    }


def detect(b: Bars, cfg: dict, pattern: str) -> list[dict]:
    """Stage events of one ticker for ``pattern`` (CB or AT), deduplicated per first touch."""
    x = _Ctx(b, cfg)
    cand = [e for jj in range(len(x.peaks)) for e in _scan(x, jj, pattern, cfg)]
    rank = {"forming": 0, "triggered": 1}
    cand.sort(key=lambda e: (e["row"], rank[e["stage"]], e["p3_row"]))
    top: dict[int, int] = {}
    out = []
    for e in cand:
        if rank[e["stage"]] > top.get(e["start_row"], -1):
            top[e["start_row"]] = rank[e["stage"]]
            out.append(e)
    # another instance of the same base triggering on the same day overrides a same-day forming
    trig = {(e["start_row"], e["row"]) for e in out if e["stage"] == "triggered"}
    out = [e for e in out if not (e["stage"] == "forming" and (e["start_row"], e["row"]) in trig)]
    vb = cfg["consolidation"]["volume_base"]
    for e in out:
        e["vol_ratio_t"] = volume_ratio(b.volume, e["row"], e["row"], e["row"], vb)
    return out
