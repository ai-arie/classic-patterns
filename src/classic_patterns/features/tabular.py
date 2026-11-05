"""Tabular feature matrix for pattern events (LightGBM baselines and the app).

Per event, everything known at the decision close t: pattern geometry (detector output),
trade-plan features (``patterns.features``), the stock's channels at its feature-store row and the
market context of that date. Price levels and series that trend with time (log_close,
n_eligible, the raw handle low, the consolidation resistance) are left out so the features mean
the same thing in every year.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from classic_patterns.features.arrays import FeatureStore
from classic_patterns.patterns.common import ALL_PATTERNS, STAGES
from classic_patterns.patterns.detect import CONS_COLS, CUP_COLS, HS_COLS
from classic_patterns.patterns.features import OVERHEAD_COLS, trade_features

# cup / H&S geometry; the B4 pilot hybrid models were trained on exactly these columns
CORE_GEOMETRY = (
    "trigger_slope",
    "vol_ratio_t",
    *(c for c in CUP_COLS if c != "handle_low"),
    *HS_COLS,
)
CONS_GEOMETRY = tuple(c for c in CONS_COLS if c != "resistance")
# overhead resistance of long setups (prior 12 / 6-month high vs the close and the target)
GEOMETRY = (*CORE_GEOMETRY, *CONS_GEOMETRY, *OVERHEAD_COLS)
# feature sets with the optional extra block (fundamentals, earnings timing, sector strength)
EDGAR_SETS = {
    "hand+fund": ("fund", "earn"),
    "hand+sector": ("sector",),
    "hand+new": ("fund", "earn", "sector"),
    # robustness: missing fundamentals filled with the fold's training medians and the coverage
    # columns dropped, so the model cannot learn from missingness (CIK changes, see C2 report)
    "hand+new_imp": ("fund", "earn", "sector"),
}
EDGAR_IMPUTE_DROP = ("has_fund", "fund_age_days")
DROP_STOCK = ("log_close",)
DROP_MARKET = ("n_eligible",)


def tabular_features(
    events: pd.DataFrame, store: FeatureStore, close: np.ndarray, atr: np.ndarray
) -> pd.DataFrame:
    """``events``: rows of the events table (fs_row, pattern, stage, direction, stop, target,
    trigger_price and the geometry columns); ``close`` / ``atr`` at each event's row."""
    g = events["fs_row"].to_numpy(dtype=np.int64)
    # columns are collected first and the frame is built once (inserting ~100 columns one by
    # one fragments the DataFrame and makes pandas warn on every call)
    cols = {
        "pattern_code": events["pattern"].map({p: i for i, p in enumerate(ALL_PATTERNS)}),
        "stage_code": events["stage"].map({s: i for i, s in enumerate(STAGES)}),
    }
    cols = {k: v.astype(np.int8).to_numpy() for k, v in cols.items()}
    for c in GEOMETRY:
        cols[c] = events[c].to_numpy(dtype=np.float32)
    tf = trade_features(
        events["direction"], close, atr, events["stop"], events["target"], events["trigger_price"]
    )
    for c in tf.columns:
        cols[c] = tf[c].to_numpy(dtype=np.float32)
    keep = [i for i, c in enumerate(store.channels) if c not in DROP_STOCK]
    order = np.argsort(g)
    x = np.empty((len(g), len(keep)), dtype=np.float32)
    x[order] = np.asarray(store.x[g[order]])[:, keep]
    for j, i in enumerate(keep):
        cols[store.channels[i]] = x[:, j]
    ctx = store.rows["ctx_idx"].to_numpy()[g]
    mk = store.market.drop(columns=["date", *DROP_MARKET])
    m = mk.to_numpy(dtype=np.float32)[ctx]
    for j, c in enumerate(mk.columns):
        cols[f"mkt_{c}"] = m[:, j]
    return pd.DataFrame(cols, index=events.index)
