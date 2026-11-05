"""Hand-labeling helpers: CSV upsert and the stratified sample."""

import numpy as np
import pandas as pd

from classic_patterns.patterns.common import PATTERNS, STAGES
from classic_patterns.patterns.labeling import (
    draw_sample,
    load_labels,
    refresh_sample,
    save_label,
    top_up_sample,
)
from classic_patterns.utils.config import load_config


def test_save_label_upserts(tmp_path):
    p = tmp_path / "labels" / "human_checks.csv"
    assert load_labels(p).empty
    save_label(p, {"sample_id": 2, "event_id": "b", "label": "valid"})
    save_label(p, {"sample_id": 1, "event_id": "a", "label": "invalid", "reasons": "noisy;other"})
    df = save_label(p, {"sample_id": 2, "event_id": "b", "label": "unsure"})
    assert list(df["event_id"]) == ["a", "b"] and list(df["label"]) == ["invalid", "unsure"]
    back = load_labels(p)
    assert back.equals(df.reset_index(drop=True)) or len(back) == 2
    assert back.loc[back["event_id"] == "a", "reasons"].item() == "noisy;other"


def test_draw_sample_stratified_and_pre_holdout():
    cfg = load_config("detector")
    cfg["sample"]["per_cell"] = 4
    rng = np.random.default_rng(0)
    n = 4000
    ev = pd.DataFrame(
        {
            "pattern": rng.choice(PATTERNS, n),
            "stage": rng.choice(STAGES, n),
            "date": pd.to_datetime("1996-01-01") + pd.to_timedelta(rng.integers(0, 11000, n), "D"),
            "ticker": rng.choice([f"T{i}" for i in range(300)], n),
            "elig": rng.choice([0, 1, 7], n).astype(np.uint8),
            "data_flag": rng.random(n) < 0.05,
            "failed_default": rng.choice(["", "", "cup_days", "time_sym;neck_slope"], n),
        }
    )
    ev["pattern_id"] = [f"p{i % 2500}" for i in range(n)]
    s = draw_sample(ev, cfg)
    assert (s["date"] <= cfg["sample"]["end"]).all()
    assert ((s["elig"] & 1) == 1).all() and not s["data_flag"].any()
    assert s["pattern_id"].is_unique
    cells = s.groupby(["pattern", "stage", "borderline"]).size()
    assert len(cells) == len(PATTERNS) * len(STAGES) * 2 and (cells == 2).all()
    assert list(s["sample_id"]) == list(range(1, len(s) + 1))


def test_refresh_sample_keeps_events_and_order():
    ev = pd.DataFrame(
        {
            "pattern": ["cup_handle", "hs_top", "inv_hs"],
            "event_id": ["a", "b", "c"],
            "failed_default": ["", "", "time_sym"],
            "x": [1.0, 2.0, 3.0],
        }
    )
    old = pd.DataFrame({"sample_id": [1, 2], "event_id": ["c", "a"]})
    new = ev.assign(failed_default=["handle_days", "", ""], x=[10.0, 20.0, 30.0])
    s = refresh_sample(old, new)
    assert list(s["sample_id"]) == [1, 2] and list(s["event_id"]) == ["c", "a"]
    assert list(s["x"]) == [30.0, 10.0] and list(s["borderline"]) == [False, True]
    assert s["detected"].all()
    # an event no longer detected keeps its previous row and is flagged
    old_full = s.drop(columns=["borderline", "detected"])
    s2 = refresh_sample(old_full, new[new["event_id"] != "a"].assign(x=[20.0, 99.0]))
    assert list(s2["event_id"]) == ["c", "a"] and list(s2["detected"]) == [True, False]
    assert list(s2["x"]) == [99.0, 10.0]


def test_top_up_appends_after_sample_and_is_idempotent():
    cfg = load_config("detector")
    cfg["sample"]["top_up"] = [
        {"pattern": "cup_handle", "stage": "forming", "borderline": False, "n": 3}
    ]
    n = 40
    loose = pd.DataFrame(
        {
            "pattern": ["cup_handle"] * n,
            "stage": ["forming"] * n,
            "date": pd.to_datetime("2000-01-03") + pd.to_timedelta(np.arange(n), "D"),
            "ticker": [f"T{i}" for i in range(n)],
            "elig": np.ones(n, dtype=np.uint8),
            "data_flag": np.zeros(n, dtype=bool),
            "failed_default": ["", "handle_days"] * (n // 2),
            "pattern_id": [f"p{i}" for i in range(n)],
        }
    )
    base = loose.iloc[:2].assign(borderline=[False, True], top_up=False)
    base.insert(0, "sample_id", [1, 2])
    s = top_up_sample(base, loose, cfg)
    add = s[s["top_up"]]
    assert list(s["sample_id"]) == [1, 2, 3, 4, 5]
    assert len(add) == 3 and not add["borderline"].any()
    assert not add["pattern_id"].isin(base["pattern_id"]).any()
    assert top_up_sample(s, loose, cfg).equals(s)


def test_top_up_entries_counted_separately_with_query():
    cfg = load_config("detector")
    cfg["sample"]["top_up"] = [
        {"pattern": "cup_handle", "stage": "forming", "borderline": False, "n": 2},
        {"pattern": "cup_handle", "stage": "forming", "query": "x >= 30", "n": 3},
    ]
    n = 40
    loose = pd.DataFrame(
        {
            "pattern": ["cup_handle"] * n,
            "stage": ["forming"] * n,
            "date": pd.to_datetime("2000-01-03") + pd.to_timedelta(np.arange(n), "D"),
            "ticker": [f"T{i}" for i in range(n)],
            "elig": np.ones(n, dtype=np.uint8),
            "data_flag": np.zeros(n, dtype=bool),
            "failed_default": ["", "handle_days"] * (n // 2),
            "pattern_id": [f"p{i}" for i in range(n)],
            "x": np.arange(n),
        }
    )
    # an older sample: one top-up row and no top_up_id column -> counted as entry 0
    base = loose.iloc[[0, 2]].assign(borderline=False, top_up=[False, True])
    base.insert(0, "sample_id", [1, 2])
    s = top_up_sample(base, loose, cfg)
    assert list(s["top_up_id"]) == [-1, 0, 0, 1, 1, 1]
    assert (s.loc[s["top_up_id"] == 1, "x"] >= 30).all()
    assert top_up_sample(s, loose, cfg).equals(s)
