"""Head-and-shoulders top (short) and inverse head-and-shoulders (long): one implementation with a
shape sign ``s`` (+1: peaks are highs, troughs are lows; -1: the mirror image).

Pivots come from the zigzag at each scale in ``pivots.zigzag_atr``. For a zigzag peak H with the
two previous pivots LS (peak) and T1 (trough), and the first trough T2 after H:

- ``forming``: RS (most extreme bar since T2) confirmed by a zigzag-sized reversal, all gates pass,
  close within ``near_neckline`` of the neckline but not through it.
- ``triggered``: first close through the neckline (line through T1 and T2, projected to day t),
  all gates pass. Entry is the next open.

T2 is re-taken as the most extreme trough between H and RS when RS moves. Each scale is detected
on its own; ``detect`` merges them per head in time order (ties: smaller scale) and keeps an
event only if its stage is later than every stage already emitted for that head. Both choices
only use bars <= the event day, so the output is truncation-invariant.
"""

from __future__ import annotations

import numpy as np

from classic_patterns.patterns.common import Bars, at_least, at_most, in_range, volume_ratio
from classic_patterns.patterns.pivots import clean_pivot_bars, zigzag

NAMES = {1: "hs_top", -1: "inv_hs"}


def gate_failures(g: dict, c: dict, stage: str) -> list[str]:
    f = []
    if not at_least(g["prior_trend"], c["prior_trend_min"]):
        f.append("prior_trend")
    tol = c["head_extreme_tol"]
    if tol is not None and not at_least(g["head_vs_extreme"], -tol):
        f.append("head_vs_extreme")
    if not at_least(g["head_prominence"], c["head_prominence_min"]):
        f.append("head_prominence")
    if not at_least(g["head_prominence_atr"], c["head_prominence_atr_min"]):
        f.append("head_prominence_atr")
    if not at_least(g["head_prominence_close_atr"], c["head_prominence_close_atr_min"]):
        f.append("head_prominence_close_atr")
    if not in_range(g["ls_height"], c["shoulder_height"]):
        f.append("shoulder_height")
    if not in_range(g["rs_height"], c["shoulder_height"]) and "shoulder_height" not in f:
        f.append("shoulder_height")
    if not at_most(g["shoulder_diff"], c["shoulder_diff_max"]):
        f.append("shoulder_diff")
    if not at_most(g["shoulder_diff_hh"], c["shoulder_diff_hh_max"]):
        f.append("shoulder_diff_hh")
    if not in_range(g["time_sym"], c["time_sym"]):
        f.append("time_sym")
    if not at_most(abs(g["neck_slope"]), c["neck_slope_max"]):
        f.append("neck_slope")
    if not in_range(g["span_days"], c["span_days"]):
        f.append("span_days")
    if not at_least(g["min_leg_atr"], c.get("min_leg_atr")):
        f.append("min_leg")
    if not at_least(g["min_spacing"], c.get("min_spacing")):
        f.append("pivot_spacing")
    if not at_most(g["spike_pivots"], c.get("spike_pivots_max")):
        f.append("spike_pivot")
    return f


class _Shape:
    """Price arrays seen from the pattern's side: P = peaks, Q = troughs."""

    def __init__(self, b: Bars, s: int):
        self.s = s
        self.P = b.high if s == 1 else b.low
        self.Q = b.low if s == 1 else b.high

    def rel(self, a: float, ref: float) -> float:
        return self.s * (a - ref) / ref

    def ext_idx(self, x: np.ndarray, lo: int, hi: int) -> int:
        """Row of the most extreme value of x (peak side) in rows [lo, hi]."""
        seg = x[lo : hi + 1]
        return lo + int(np.argmax(seg) if self.s == 1 else np.argmin(seg))

    def trough_idx(self, lo: int, hi: int) -> int:
        seg = self.Q[lo : hi + 1]
        return lo + int(np.argmin(seg) if self.s == 1 else np.argmax(seg))


def _left(b: Bars, sh: _Shape, LS: int, T1: int, H: int, t: int, c: dict) -> dict:
    s, P, Q = sh.s, sh.P, sh.Q
    w = Q[max(0, LS - c["prior_window"]) : LS]
    ref = (w.min() if s == 1 else w.max()) if len(w) else np.nan
    we = P[max(0, H - c["head_extreme_window"]) : H]
    ext = (we.max() if s == 1 else we.min()) if len(we) else np.nan
    hw, vb = c["volume_halfwidth"], c["volume_base"]
    return {
        "prior_trend": sh.rel(P[LS], ref),
        "head_vs_extreme": sh.rel(P[H], ext),
        "left_days": H - LS,
        "head_prominence": sh.rel(P[H], P[LS]),
        "ls_height": (P[LS] - Q[T1]) / (P[H] - Q[T1]),
        "vol_ls": volume_ratio(b.volume, LS - hw, min(LS + hw, t), LS, vb),
        "vol_h": volume_ratio(b.volume, H - hw, min(H + hw, t), LS, vb),
    }


def _full(b, sh, LS, T1, H, T2, RS, t, left, c, clean) -> dict:
    s, P, Q = sh.s, sh.P, sh.Q
    m = (Q[T2] - Q[T1]) / (T2 - T1)

    def nl(x):
        return Q[T1] + m * (x - T1)

    hh = s * (P[H] - nl(H))
    sh_ext = P[LS] if s * (P[LS] - P[RS]) >= 0 else P[RS]
    hw, vb, cw = c["volume_halfwidth"], c["volume_base"], c["close_halfwidth"]

    def ext_close(i):  # most extreme close within cw bars of pivot i, not past t
        seg = b.close[max(0, i - cw) : min(i + cw, t) + 1]
        return seg.max() if s == 1 else seg.min()

    cs = (max if s == 1 else min)(ext_close(LS), ext_close(RS))
    return {
        **left,
        "head_prominence": sh.rel(P[H], sh_ext),
        "head_prominence_atr": s * (P[H] - sh_ext) / b.atr[H],
        "head_prominence_close_atr": s * (ext_close(H) - cs) / b.atr[H],
        "head_height": hh / nl(H),
        "ls_height": s * (P[LS] - nl(LS)) / hh,
        "rs_height": s * (P[RS] - nl(RS)) / hh,
        "shoulder_diff": abs(P[LS] - P[RS]) / sh_ext,
        "shoulder_diff_hh": abs(P[LS] - P[RS]) / hh,
        "time_sym": (RS - H) / (H - LS),
        "neck_slope": s * (nl(RS) - nl(LS)) / hh,
        "span_days": RS - LS,
        # each leg LS-T1-H-T2-RS in ATR14 at the head; bars between neighbouring pivots;
        # pivots on spike / gap bars
        "min_leg_atr": min(
            s * (P[LS] - Q[T1]), s * (P[H] - Q[T1]), s * (P[H] - Q[T2]), s * (P[RS] - Q[T2])
        )
        / b.atr[H],  # fmt: skip
        "min_spacing": float(min(T1 - LS, H - T1, T2 - H, RS - T2)),
        "spike_pivots": float(sum(not clean[i] for i in (LS, T1, H, T2, RS))),
        "vol_rs": volume_ratio(b.volume, RS - hw, min(RS + hw, t), LS, vb),
        "neckline": nl(t),
        "neck_m": m,
        "head_px": P[H] - nl(H),
    }


def _scale_events(b: Bars, s: int, mult: float, c: dict, clean: np.ndarray) -> list[dict]:
    sh = _Shape(b, s)
    P, Q = sh.P, sh.Q
    n = len(b)
    thr = mult * b.atr
    z = zigzag(b.high, b.low, thr)
    tmin, tmax = c["time_sym"]
    out: list[dict] = []
    for j in np.flatnonzero(z.kind == s):
        if j < 2:
            continue
        LS, T1, H, cH = (int(z.idx[j - 2]), int(z.idx[j - 1]), int(z.idx[j]), int(z.confirm[j]))
        left = _left(b, sh, LS, T1, H, cH, c)
        base = {"LS": LS, "T1": T1, "H": H, "scale": mult}
        if j + 1 >= len(z.idx):
            continue
        T2c, cT2 = int(z.idx[j + 1]), int(z.confirm[j + 1])
        last = min(n - 1, H + int(c["expiry_ratio"] * (H - LS)))
        RS = sh.ext_idx(P, T2c + 1, cT2)
        formed = False
        for t in range(cT2, last + 1):
            if t > cT2 and s * (P[t] - P[RS]) > 0:
                RS = t
            if sh.rel(P[H], P[RS]) < c["head_prominence_min"] or (RS - H) / (H - LS) > tmax:
                break  # RS too extreme or too late: can only get worse
            if RS == t:
                continue
            T2 = sh.trough_idx(H + 1, RS - 1)
            g = _full(b, sh, LS, T1, H, T2, RS, t, left, c, clean)
            if not (g["head_px"] * s > 0 and g["neckline"] > 0 and g["head_height"] > 0):
                break  # T2 beyond the head, or a falling neckline projected to <= 0
            ok = not gate_failures(g, c, "triggered")
            ev = {**base, "T2": T2, "RS": RS}
            if s * (b.close[t] - g["neckline"]) < 0:
                if ok:
                    out.append(_event(s, "triggered", t, ev, g, P[RS]))
                break
            rs_conf = s * (P[RS] - Q[sh.trough_idx(RS + 1, t)]) >= thr[t]
            near = s * (b.close[t] - g["neckline"]) / g["neckline"] <= c["near_neckline"]
            if not formed and ok and rs_conf and near:
                out.append(_event(s, "forming", t, ev, g, P[RS]))
                formed = True
    return out


def _event(s: int, stage: str, t: int, p: dict, g: dict, stop: float) -> dict:
    name = NAMES[s]
    return {
        "pattern": name,
        "stage": stage,
        "row": t,
        "start_row": p["LS"],
        "head_row": p["H"],
        "p1_row": p["LS"],
        "p2_row": p["T1"],
        "p3_row": p["H"],
        "p4_row": p["T2"],
        "p5_row": p["RS"],
        "trigger_price": g["neckline"],
        "trigger_slope": g["neck_m"],
        "stop": stop,
        "target": g["neckline"] - g["head_px"],
        "pid": f"{name}:{p['H']}",
        "scale": p["scale"],
        **{k: v for k, v in g.items() if k not in ("neckline", "neck_m", "head_px")},
    }


def detect(b: Bars, cfg: dict, s: int) -> list[dict]:
    """Stage events for one ticker and shape (``s`` = 1 top, -1 inverse), deduplicated per head."""
    c = cfg["head_shoulders"]
    rank = {"forming": 0, "triggered": 1}
    clean = clean_pivot_bars(b, c.get("spike_atr"), c.get("gap_atr"))
    cand = [e for mult in cfg["pivots"]["zigzag_atr"] for e in _scale_events(b, s, mult, c, clean)]
    cand.sort(key=lambda e: (e["row"], rank[e["stage"]], e["scale"]))
    top: dict[int, int] = {}  # head row -> highest stage emitted so far
    out = []
    for e in cand:
        if rank[e["stage"]] > top.get(e["head_row"], -1):
            top[e["head_row"]] = rank[e["stage"]]
            out.append(e)
    # another scale triggering the same head on the same day overrides a same-day 'forming'
    trig = {(e["head_row"], e["row"]) for e in out if e["stage"] == "triggered"}
    out = [e for e in out if not (e["stage"] == "forming" and (e["head_row"], e["row"]) in trig)]
    for e in out:
        e["vol_ratio_t"] = volume_ratio(b.volume, e["row"], e["row"], e["row"], c["volume_base"])
    return out
