"""Consolidation breakout and ascending triangle detectors on designed paths."""

import re

import numpy as np
import pytest
import synthetic

from classic_patterns.patterns.common import Bars
from classic_patterns.patterns.consolidation import touch_levels
from classic_patterns.patterns.detect import detect_ticker
from classic_patterns.patterns.labeling import REASONS_AT, REASONS_CB
from classic_patterns.patterns.pivots import clean_pivot_bars
from classic_patterns.utils.code_hash import project_root
from classic_patterns.utils.config import load_config

CFG = load_config("detector")


def _one(ev, pattern, stage):
    sel = ev[(ev["pattern"] == pattern) & (ev["stage"] == stage)]
    assert len(sel) == 1, sel[["pattern", "stage", "date", "start_row"]]
    return sel.iloc[0]


def test_flat_base_breakout():
    path, p = synthetic.base_path()
    df = synthetic.from_path(path)
    ev = detect_ticker(df, CFG)
    e = _one(ev, "consolidation_breakout", "triggered")
    assert e["row"] == p["trigger"] and e["direction"] == 1
    # the last return to the resistance runs straight into the breakout: not a swing high
    assert e["p1_row"] == p["touches"][0] and e["touches"] == len(p["touches"]) - 1
    assert np.isclose(e["resistance"], df["high"].iloc[p["touches"][0]])
    assert e["flatness_atr"] < 1e-9 and e["subtype"] == "flat_base"
    assert np.isclose(
        e["target"], 2 * e["resistance"] - df["low"].iloc[p["touches"][0] : e["row"]].min()
    )
    assert e["stop"] < e["trigger_price"] < e["target"]
    f = _one(ev, "consolidation_breakout", "forming")
    assert f["row"] < e["row"]
    # flat lows are not an ascending triangle
    assert not (ev["pattern"] == "ascending_triangle").any()


def test_falling_lows_fail_default_but_not_loose():
    path, _ = synthetic.base_path(lows=(95.0, 90.0, 85.0))
    df = synthetic.from_path(path)
    assert not (detect_ticker(df, CFG)["pattern"] == "consolidation_breakout").any()
    lo = detect_ticker(df, CFG, loose=True)
    t = lo[(lo["pattern"] == "consolidation_breakout") & (lo["stage"] == "triggered")]
    assert len(t) and t["failed_default"].str.contains("lows_trend").all()


def test_base_too_deep_in_atr_fails_default_but_not_loose():
    df = synthetic.from_path(synthetic.base_path(lows=(85.0, 85.0, 85.0))[0])
    assert not (detect_ticker(df, CFG)["pattern"] == "consolidation_breakout").any()
    lo = detect_ticker(df, CFG, loose=True)
    t = lo[(lo["pattern"] == "consolidation_breakout") & (lo["stage"] == "triggered")]
    assert len(t) and (t["height_atr"] > CFG["consolidation_breakout"]["height_max_atr"]).all()
    assert t["failed_default"].str.contains("height_max").all()


def test_ascending_triangle_up_and_down():
    for down in (False, True):
        path, p = synthetic.triangle_path(down)
        df = synthetic.from_path(path)
        ev = detect_ticker(df, CFG)
        e = _one(ev, "ascending_triangle", "triggered")
        assert e["p1_row"] == p["first"] and e["p2_row"] == p["lows"][0]
        assert e["direction"] == (-1 if down else 1) and e["breakout_dir"] == e["direction"]
        assert e["n_lows"] >= 2 and e["lows_rise_min_atr"] >= 0.5 and 0 < e["breakout_pos"] <= 1
        if down:
            lo = df["low"].to_numpy()
            s1 = p["lows"][0]
            m = min((lo[q] - lo[s1]) / (q - s1) for q in p["lows"][1:])  # lowest line
            sup = lo[s1] + m * (e["row"] - s1)
            assert np.isclose(e["trigger_price"], sup) and df["close"].iloc[e["row"]] < sup
            assert e["stop"] == e["resistance"] and e["target"] < e["trigger_price"]
        else:
            assert e["target"] > e["trigger_price"] == e["resistance"] > e["stop"]


def test_spikes_and_gaps_are_not_pivots_and_wicks_are_capped():
    df = synthetic.from_path(synthetic.base_path()[0])
    b = Bars.from_frame(df)
    b.high, b.open = b.high.copy(), b.open.copy()
    i = 480
    b.high[i] = b.close[i] + 10 * b.atr[i]  # spike bar
    b.open[i + 5] = b.close[i + 4] + 3 * b.atr[i + 5]  # gap bar
    ok = clean_pivot_bars(b, 3.0, 1.0)
    assert not ok[i] and not ok[i + 5] and ok[i + 2]
    b.high[i + 2] = max(b.open[i + 2], b.close[i + 2]) + 2 * b.atr[i + 2]  # long wick
    lv = touch_levels(b, {"wick_cap_atr": 0.5})
    assert np.isclose(lv[i + 2], max(b.open[i + 2], b.close[i + 2]) + 0.5 * b.atr[i + 2])


def test_reason_codes_match_consolidation_sheet():
    sheet = project_root() / "docs" / "labeling" / "consolidation_breakouts.md"
    if not sheet.exists():
        pytest.skip("labeling sheets not in this copy")
    text = sheet.read_text()
    secs = text.split('### Reason codes for "invalid"')[1:]
    codes = [
        tuple(re.findall(r"^\| `([a-z_]+)` \|", re.split(r"\n---\n|\n## ", s)[0], re.M))
        for s in secs
    ]
    assert codes == [REASONS_CB, REASONS_AT]


def test_forming_never_on_or_after_trigger_real_data():
    import pandas as pd

    prices = project_root() / "data" / "clean" / "prices.parquet"
    if not prices.exists():
        import pytest

        pytest.skip("data/clean not available")
    df = pd.read_parquet(prices, columns=["ticker", "date", "open", "high", "low", "close", "volume"],
                         filters=[("ticker", "in", ["HST", "KMB", "TMO", "NTRS", "KSS"])])  # fmt: skip
    for _, g in df.groupby("ticker"):
        ev = detect_ticker(g.sort_values("date").reset_index(drop=True), CFG)
        ev = ev[ev["pattern"].isin(["consolidation_breakout", "ascending_triangle"])]
        f = ev[ev["stage"] == "forming"].set_index(["pattern", "start_row"])["row"]
        t = ev[ev["stage"] == "triggered"].set_index(["pattern", "start_row"])["row"]
        j = pd.concat([f.rename("f"), t.rename("t")], axis=1, join="inner")
        assert (j["t"] > j["f"]).all()


def _triangle_below_older_high() -> np.ndarray:
    """triangle_path with an older high at 104 between the triangle top (100) and its target."""
    path, _ = synthetic.triangle_path()
    warm = 50 + 0.2 * np.sin(np.arange(300) / 3)
    pre = [warm, synthetic._seg(warm[-1], 104, 120), synthetic._seg(104, 92, 15),
           synthetic._seg(92, 100, 15)]  # fmt: skip
    return np.concatenate([*pre, path[300 + 150 :]])


def test_triangle_with_older_high_in_path_flagged_and_optional_gate():
    df = synthetic.from_path(_triangle_below_older_high())
    # default: detected, with the older high in the path in its overhead columns
    e = _one(detect_ticker(df, CFG), "ascending_triangle", "triggered")
    assert e["overhead252_atr"] > 0.5 and 0 < e["overhead252_frac"] < 1
    # with the optional gate switched on: rejected by default, kept and flagged in the loose build
    gate = {"window": 252, "clear_atr": 0.5}
    cfg = {**CFG, "ascending_triangle": {**CFG["ascending_triangle"], "overhead_gate": gate}}
    ev = detect_ticker(df, cfg)
    assert not ((ev["pattern"] == "ascending_triangle") & (ev["stage"] == "triggered")).any()
    lo = detect_ticker(df, cfg, loose=True)
    t = lo[(lo["pattern"] == "ascending_triangle") & (lo["stage"] == "triggered")]
    assert len(t) and t["failed_default"].str.contains("overhead").all()


def test_overhead_columns_clear_path_and_short_setups():
    for down in (False, True):
        ev = detect_ticker(synthetic.from_path(synthetic.triangle_path(down)[0]), CFG)
        e = _one(ev, "ascending_triangle", "triggered")
        if down:
            assert np.isnan(e["overhead252_atr"]) and np.isnan(e["overhead126_frac"])
        else:  # breakout at new highs: the prior high is at most the triangle top
            assert e["overhead252_atr"] <= 0.5 and e["overhead252_frac"] < 0.1
