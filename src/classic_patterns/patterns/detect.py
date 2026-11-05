"""Run all detectors on one ticker: the single detection code path for training, evaluation and the
app. Returns one row per (pattern instance, stage), each dated on the day the stage is reached."""

from __future__ import annotations

import numpy as np
import pandas as pd

from classic_patterns.patterns import consolidation, cup_handle, head_shoulders
from classic_patterns.patterns.common import CUP_PATTERNS, DIRECTION, STAGES, Bars, merged
from classic_patterns.patterns.features import OVERHEAD_COLS, event_overhead

ROW_COLS = ("row", "start_row", "head_row", "p1_row", "p2_row", "p3_row", "p4_row", "p5_row")
LEVEL_COLS = ("trigger_price", "trigger_slope", "stop", "target", "vol_ratio_t")
CUP_COLS = (
    "prior_rise",
    "lip_vs_high",
    "cup_days",
    "cup_depth",
    "lip_diff",
    "bottom_frac",
    "cup_r2",
    "cup_low_pos",
    "cup_vix_pct_max",
    "cup_low_vix_pct",
    "cup_vol_ratio",
    "handle_low",
    "handle_days",
    "handle_cup_ratio",
    "handle_depth",
    "handle_pos",
    "handle_slope",
    "handle_vol_ratio",
    "lip_recovery",
    "cup_low_vs_trend",
    "spike_lips",
)
HS_COLS = (
    "scale",
    "prior_trend",
    "head_vs_extreme",
    "left_days",
    "head_prominence",
    "head_prominence_atr",
    "head_prominence_close_atr",
    "head_height",
    "ls_height",
    "rs_height",
    "shoulder_diff",
    "shoulder_diff_hh",
    "time_sym",
    "neck_slope",
    "span_days",
    "vol_ls",
    "vol_h",
    "vol_rs",
    "min_leg_atr",
    "min_spacing",
    "spike_pivots",
)
CONS_COLS = (
    "touches",
    "flatness_atr",
    "res_vs_high",
    "duration_days",
    "base_depth",
    "height_atr",
    "n_lows",
    "lows_trend_atr",
    "lows_rise_min_atr",
    "closes_out",
    "tightness",
    "base_vol_ratio",
    "prior_drop",
    "breakout_pos",
    "support_slope_atr",
    "subtype_code",
    "breakout_dir",
    "gap_atr_t",
    "resistance",
)
CONS_PATTERNS = (consolidation.CB, consolidation.AT)


def gate_failures(e: dict, cfg: dict) -> list[str]:
    if e["pattern"] in CUP_PATTERNS:
        deep = e["pattern"] == cup_handle.DEEP
        return cup_handle.gate_failures(e, cfg["cup_handle"], e["stage"], deep)
    if e["pattern"] in CONS_PATTERNS:
        return consolidation.gate_failures(e, cfg[e["pattern"]], e["stage"], e["pattern"])
    return head_shoulders.gate_failures(e, cfg["head_shoulders"], e["stage"])


def detect_ticker(
    df: pd.DataFrame, cfg: dict, vix_pct: pd.Series | None = None, loose: bool = False
) -> pd.DataFrame:
    """Pattern stage events for one ticker.

    ``df``: rows sorted by date with date, open, high, low, close, volume (adjusted).
    ``vix_pct``: VIX 1-year percentile by date (allows deeper cups in high-VIX markets).
    ``loose``: use the looser labeling thresholds; adds ``failed_default`` (gates the event
    fails under the default thresholds, ';'-joined).
    """
    c = merged(cfg, loose)
    b = Bars.from_frame(df, cfg["pivots"]["atr"], vix_pct)
    ev = cup_handle.detect(b, c)
    for s in (1, -1):
        ev += head_shoulders.detect(b, c, s)
    for pat in CONS_PATTERNS:
        ev += consolidation.detect(b, c, pat)
    # overhead resistance of long setups (prior highs only); also an input of the triangle gate
    d = [e.get("trade_dir") if e.get("trade_dir") is not None else DIRECTION[e["pattern"]]
         for e in ev]  # fmt: skip
    oh = event_overhead(b.high, b.close, b.atr, [e["row"] for e in ev], d,
                        [e["target"] for e in ev])  # fmt: skip
    for e, vals in zip(ev, oh.to_dict("records"), strict=True):
        e.update(vals)
    if loose:
        for e in ev:
            e["failed_default"] = ";".join(gate_failures(e, cfg))
    extra = ["trade_dir", "subtype"]
    cols = ["pattern", "stage", "pid", *ROW_COLS, *LEVEL_COLS, *CUP_COLS, *HS_COLS, *CONS_COLS,
            *OVERHEAD_COLS]  # fmt: skip
    cols += extra
    if loose:
        cols.append("failed_default")
    out = pd.DataFrame(ev, columns=cols)
    num = [*LEVEL_COLS, *CUP_COLS, *HS_COLS, *CONS_COLS, *OVERHEAD_COLS, "trade_dir"]
    out[num] = out[num].astype(np.float64)
    out[list(ROW_COLS)] = out[list(ROW_COLS)].astype(np.int64)
    txt = ["pattern", "stage", "pid", *(["failed_default"] if loose else [])]
    out[txt] = out[txt].astype("str")
    out["subtype"] = out["subtype"].astype("string")
    # trade direction: the pattern's, or the event's own (ascending triangle: breakout side)
    d = out["trade_dir"].fillna(out["pattern"].map(DIRECTION))
    out["direction"] = d.astype(np.int8)
    out = out.drop(columns="trade_dir")
    out["stage_n"] = out["stage"].map({s: i for i, s in enumerate(STAGES)})
    out = out.sort_values(["row", "pattern", "pid", "stage_n"], kind="stable")
    out = out.drop(columns="stage_n").reset_index(drop=True)
    dates = b.dates
    for col in ROW_COLS:
        r = out[col].to_numpy()
        name = "date" if col == "row" else col.replace("_row", "_date")
        out[name] = np.where(r >= 0, dates[np.clip(r, 0, None)], np.datetime64("NaT"))
        out[name] = out[name].astype("datetime64[ns]")
    return out
