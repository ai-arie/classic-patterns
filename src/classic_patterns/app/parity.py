"""Match the bundle's parity sample against locally detected setups and judge the score differences.

Yahoo re-adjusts a stock's whole price history after every dividend or split, so the absolute stop
and target of an old setup change while its shape does not. The sample therefore holds no prices.
Setups are matched on (ticker, date, pattern, stage, direction) and on scale-free geometry (planned
reward/risk, risk in ATR units), and every reference setup gets a status:

- ``matched``: same geometry as on the training machine.
- ``changed``: the setup is still detected but its geometry differs (revised price data).
- ``missing``: not detected locally any more.

``matched`` setups are compared. The model inputs are scale-free, so a correct installation
reproduces their scores; small differences come from data that changed since the bundle was built
(revised prices or volumes, a different universe moving market breadth).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

KEY = ["ticker", "date", "pattern", "stage", "direction"]
GEOM = ["rr_planned", "risk_atr"]
MODELS = ("lgbm", "student")
CLASSES = ("target", "stop", "timeout")
COMPARED = ("matched",)

TOL = {"lgbm": 1e-6, "student": 0.02}  # same data -> same scores (int8 student: CPU-dependent)
DRIFT_TOL = {"lgbm": 0.02, "student": 0.05}  # largest difference accepted as data drift
GEOM_TOL = 0.01  # max |difference| of rr_planned / risk_atr for the same setup
MIN_FOUND = 0.95  # share of reference setups that must be compared
MIN_WITHIN = 0.95  # share of compared setups that must be within TOL


def match(ref: pd.DataFrame, cand: pd.DataFrame, geom_tol: float = GEOM_TOL) -> pd.DataFrame:
    """One row per reference setup with its status and score differences.

    ``ref``: the parity sample (KEY, GEOM, ``{model}_p_{class}``).
    ``cand``: locally scored candidates with the same columns plus ``ref_i``, the row of ``ref``
    whose KEY they share (a setup can be found at several zigzag scales: several candidates; the
    closest geometry is used).
    """
    ref = ref.reset_index(drop=True)
    r = ref.loc[cand["ref_i"].to_numpy()].reset_index(drop=True)
    c = cand.reset_index(drop=True)
    d_geom = np.max(np.abs(c[GEOM].to_numpy(float) - r[GEOM].to_numpy(float)), axis=1)
    c = c.assign(d_geom=d_geom, status=np.where(d_geom > geom_tol, "changed", "matched"))
    for m in MODELS:
        cols = [f"{m}_p_{k}" for k in CLASSES]
        c[f"d_{m}"] = np.abs(c[cols].to_numpy(float) - r[cols].to_numpy(float)).max(axis=1)
    best = c.sort_values(["ref_i", "d_geom"]).drop_duplicates("ref_i").set_index("ref_i")
    out = ref[KEY].copy()
    out["status"] = best["status"].reindex(out.index).fillna("missing")
    for col in ("d_geom", "d_lgbm", "d_student"):
        out[col] = best[col].reindex(out.index)
    out.loc[~out["status"].isin(COMPARED), ["d_lgbm", "d_student"]] = np.nan
    return out


def verdict(
    m: pd.DataFrame,
    tol: dict = TOL,
    drift_tol: dict = DRIFT_TOL,
    min_found: float = MIN_FOUND,
    min_within: float = MIN_WITHIN,
) -> tuple[str, list[str]]:
    """``OK`` (everything reproduced), ``DRIFT`` (small differences explained by changed data) or
    ``FAILED``, plus the reasons for anything short of ``OK``."""
    comp = m[m["status"].isin(COMPARED)]
    within = np.ones(len(comp), bool)
    for k in MODELS:
        within &= comp[f"d_{k}"].to_numpy() <= tol[k]
    if len(comp) == len(m) and within.all():
        return "OK", []
    why = []
    found = len(comp) / max(len(m), 1)
    if found < min_found:
        why.append(f"only {found:.1%} of reference setups compared (need {min_found:.0%})")
    share = within.mean() if len(comp) else 0.0
    if share < min_within:
        why.append(f"only {share:.1%} of compared setups within tolerance (need {min_within:.0%})")
    for k in MODELS:
        worst = comp[f"d_{k}"].max() if len(comp) else np.nan
        if not worst <= drift_tol[k]:
            why.append(f"{k}: max |dp| {worst:.2e} above the data-drift limit {drift_tol[k]:g}")
    return ("FAILED" if why else "DRIFT"), why
