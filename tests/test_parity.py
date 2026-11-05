"""Parity matching: re-adjusted prices are matched by geometry; verdict separates drift from bugs."""

import numpy as np
import pandas as pd

from classic_patterns.app.parity import CLASSES, MODELS, match, verdict

P = {f"{m}_p_{c}": v for m in MODELS for c, v in zip(CLASSES, (0.5, 0.4, 0.1), strict=True)}


def _ref(n):
    return pd.DataFrame(
        {
            "ticker": [f"T{i}" for i in range(n)],
            "date": pd.Timestamp("2010-01-04"),
            "pattern": "cup_handle",
            "stage": "triggered",
            "direction": 1,
            "rr_planned": 1.5,
            "risk_atr": 2.0,
            **P,
        }
    )


def _cand(ref, idx, **over):
    c = ref.loc[idx].reset_index(names="ref_i")
    for k, v in over.items():
        c[k] = v
    return c


def test_statuses_and_best_candidate():
    ref = _ref(4)
    cand = pd.concat(
        [
            _cand(ref, [0]),  # same geometry
            _cand(ref, [1], rr_planned=1.505),  # within the geometry tolerance
            _cand(ref, [2], rr_planned=2.5),  # same key, other shape
            _cand(ref, [0], rr_planned=3.0, lgbm_p_target=0.9),  # worse candidate for ref 0
        ]
    )
    m = match(ref, cand)
    assert m["status"].tolist() == ["matched", "matched", "changed", "missing"]
    assert m.loc[0, "d_lgbm"] == 0.0  # the closest candidate wins over the other scale
    assert np.isnan(m.loc[2, "d_lgbm"]) and np.isnan(m.loc[3, "d_lgbm"])


def test_verdict_ok_drift_failed():
    ref = _ref(100)
    assert verdict(match(ref, _cand(ref, list(range(100)))))[0] == "OK"

    drift = _cand(ref, list(range(98)))  # 2 missing
    drift.loc[:2, "lgbm_p_target"] += 0.005  # 3 slightly off
    assert verdict(match(ref, drift))[0] == "DRIFT"

    big = _cand(ref, list(range(100)))
    big.loc[0, "lgbm_p_target"] += 0.1  # one large difference = not data drift
    status, why = verdict(match(ref, big))
    assert status == "FAILED" and "lgbm" in why[0]

    many = _cand(ref, list(range(100)))
    many.loc[:9, "student_p_stop"] += 0.03  # 10% outside the student tolerance
    assert verdict(match(ref, many))[0] == "FAILED"

    few = _cand(ref, list(range(90)))  # 10% missing
    assert verdict(match(ref, few))[0] == "FAILED"
