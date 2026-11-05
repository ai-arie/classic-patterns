"""Screener: today's pattern setups scored with the production bundle (no PyTorch).

Setups = long rows of the events table (``scripts/detect_patterns.py --steps build``) on the
screening date, on default-eligible stock-days (close >= $5, $15M 50-day dollar volume, >= 300 days
of history), without a data-artifact flag, whose close is still between the stop and the target
(a setup whose close already reached either level has played out) and whose target is a positive
price. Short setups stay in the detectors and in the training data, but are not screened: their
results in testing were not good enough.

A triggered setup gets both models' probabilities for its textbook trade plan (entry at the next
open, the pattern's stop and target, 60 trading days). A forming setup gets only its chance of
triggering: the trade models were trained on triggered setups, and scoring a forming one from
today's close misstates its reward / risk when that close is far from the trigger.

The output describes setups; it is not a list of trades to take.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from classic_patterns.app.scoring import Bundle, close_atr, score_events
from classic_patterns.features.arrays import FeatureStore
from classic_patterns.patterns.features import OVERHEAD_COLS

log = logging.getLogger(__name__)

PATTERN_NAMES = {
    "cup_handle": "cup-and-handle",
    "deep_cup": "deep cup",
    "hs_top": "head-and-shoulders top",
    "inv_hs": "inverse head-and-shoulders",
    "consolidation_breakout": "consolidation breakout",
    "ascending_triangle": "ascending triangle",
}


TRADE_ESTIMATES = ["rr_planned", "risk_atr"] + [
    f"{m}_{c}" for m in ("lgbm", "student") for c in ("p_target", "p_stop", "p_timeout", "exp_r")
]


def result_path(out_dir: Path, day: pd.Timestamp, now: datetime | None = None) -> Path:
    """Where a screening run of session ``day`` is saved. The first run gets ``<date>.parquet`` and
    is never overwritten; a later run of the same session gets ``<date>_<local run time>.parquet``,
    which sorts after the first one, so the newest run of the latest session sorts last."""
    first = out_dir / f"{day:%Y-%m-%d}.parquet"
    if not first.exists():
        return first
    return out_dir / f"{day:%Y-%m-%d}_{now or datetime.now():%Y%m%dT%H%M%S}.parquet"


def session_of(path: Path) -> str:
    """The session date of a result file, with or without a run time in its name."""
    return path.stem.split("_")[0]


def blank_forming(df: pd.DataFrame) -> pd.DataFrame:
    """No trade estimates for setups that have not triggered."""
    out = df.copy()
    out.loc[out["stage"] == "forming", TRADE_ESTIMATES] = np.nan
    return out


def is_live(direction, close, stop, target) -> np.ndarray:
    """Close strictly between the stop and the target, and a target that is a real price."""
    d = np.sign(np.asarray(direction, dtype=np.float64))
    c, s, t = (np.asarray(x, dtype=np.float64) for x in (close, stop, target))
    return (d * (t - c) > 0) & (d * (c - s) > 0) & (t > 0)


def setups(store: FeatureStore, events: pd.DataFrame, as_of, elig_bit: int = 0) -> pd.DataFrame:
    e = events[(events["date"] == pd.Timestamp(as_of)) & (events["direction"] > 0)]
    elig = store.rows["elig"].to_numpy()[e["fs_row"].to_numpy()]
    e = e[((elig >> elig_bit) & 1).astype(bool)]
    if "data_flag" in e:
        e = e[~e["data_flag"].to_numpy(dtype=bool)]
    return e.reset_index(drop=True)


def screen(bundle: Bundle, store: FeatureStore, events: pd.DataFrame, prices: str | Path,
           as_of=None, jobs: int = 1) -> pd.DataFrame:  # fmt: skip
    """Scored setups on ``as_of`` (default: the last session of the feature build)."""
    as_of = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp(store.rows["date"].max())
    e = setups(store, events, as_of)
    if e.empty:
        return pd.DataFrame()
    close, atr = close_atr(store, prices, e["fs_row"].to_numpy())
    live = is_live(e["direction"], close, e["stop"], e["target"])
    if not live.all():
        log.info("skipped %d setups played out or with a non-positive target: %s",
                 (~live).sum(), ", ".join(e.loc[~live, "ticker"]))  # fmt: skip
    e, close, atr = e[live].reset_index(drop=True), close[live], atr[live]
    if e.empty:
        return pd.DataFrame()
    s = score_events(bundle, e, store, close, atr, jobs=jobs)
    out = pd.DataFrame(
        {
            "ticker": e["ticker"],
            "date": e["date"],
            "pattern": e["pattern"].map(PATTERN_NAMES).fillna(e["pattern"]),
            "stage": e["stage"],
            "side": np.where(e["direction"] > 0, "long", "short"),
            "close": close,
            "trigger": e["trigger_price"],
            "stop": e["stop"],
            "target": e["target"],
            "atr": atr,
            "dollar_vol50": store.rows["dollar_vol50"].to_numpy()[e["fs_row"].to_numpy()],
            "fs_row": e["fs_row"],
            "pattern_code": e["pattern"],
        }
    )
    out = blank_forming(pd.concat([out, s.reset_index(drop=True)], axis=1))
    for c in OVERHEAD_COLS:
        if c in e:
            out[c] = e[c].to_numpy()
    for c in ("start_row", "p1_row", "p2_row", "p3_row", "p4_row", "p5_row", "head_row"):
        if c in e:
            out[c] = e[c].to_numpy()
    return out
