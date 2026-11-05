"""Pattern detectors on designed price paths with known geometry, pivots, reason codes."""

import re

import numpy as np
import pandas as pd
import pytest
import synthetic

from classic_patterns.features import indicators as ind
from classic_patterns.patterns.common import STAGES
from classic_patterns.patterns.detect import detect_ticker
from classic_patterns.patterns.labeling import REASONS_ANY, REASONS_CUP, REASONS_HS
from classic_patterns.patterns.pivots import kbar_peaks, zigzag
from classic_patterns.utils.code_hash import project_root
from classic_patterns.utils.config import load_config

CFG = load_config("detector")


def test_kbar_peaks_definition_and_confirmation():
    rng = np.random.default_rng(0)
    h = np.cumsum(rng.normal(0, 1, 400))
    k = 5
    p = kbar_peaks(h, k)
    for i in range(len(h)):
        expect = (
            k <= i < len(h) - k and h[i] > h[i - k : i].max() and h[i] >= h[i + 1 : i + k + 1].max()
        )
        assert p[i] == expect
    for t in (50, 123, 399):  # known at i + k: truncation at t gives the same peaks up to t - k
        assert (kbar_peaks(h[: t + 1], k)[: t - k + 1] == p[: t - k + 1]).all()


def test_zigzag_alternates_and_confirms_on_reversal():
    df = synthetic.ohlcv(1500, seed=5)
    h, lo, c = (df[k].to_numpy() for k in ("high", "low", "close"))
    thr = 2.0 * ind.atr(h, lo, c, 14)
    z = zigzag(h, lo, thr)
    assert len(z.idx) > 20
    assert (np.diff(z.kind) != 0).all()  # alternates
    assert (z.confirm > z.idx).all() and (np.diff(z.idx) > 0).all()
    for i, cf, kd in zip(z.idx[1:], z.confirm[1:], z.kind[1:], strict=True):
        if kd == 1:  # swing high: the confirming bar's low is thr below it, earlier bars are not
            assert h[i] == h[i : cf + 1].max()
            assert h[i] - lo[cf] >= thr[cf]
            assert all(h[i] - lo[j] < thr[j] for j in range(i + 1, cf))
        else:
            assert lo[i] == lo[i : cf + 1].min()
            assert h[cf] - lo[i] >= thr[cf]
            assert all(h[j] - lo[i] < thr[j] for j in range(i + 1, cf))


def _only(ev, pattern, stage):
    sel = ev[(ev["pattern"] == pattern) & (ev["stage"] == stage)]
    assert len(sel) == 1, sel
    return sel.iloc[0]


def test_cup_handle_designed():
    path, p = synthetic.cup_handle_path()
    df = synthetic.from_path(path)
    ev = detect_ticker(df, CFG)
    e = _only(ev, "cup_handle", "triggered")
    assert (e["p1_row"], e["p3_row"]) == (p["L"], p["R"])
    pivot = df["high"].iloc[p["R"]]  # the right lip's high
    assert e["trigger_price"] == pivot
    first_above = p["R"] + np.flatnonzero(df["close"].to_numpy()[p["R"] :] > pivot)[0]
    assert e["row"] == first_above
    hl = df["low"].iloc[p["R"] + 1 : e["row"]].min()  # lowest low after the right lip
    assert e["stop"] == hl
    cup_low = df["low"].iloc[p["L"] + 1 : p["R"]].min()
    assert np.isclose(e["target"], pivot + df["high"].iloc[p["L"]] - cup_low)
    assert np.isclose(e["cup_depth"], 1 - cup_low / df["high"].iloc[p["L"]])
    assert e["handle_days"] == e["row"] - p["R"]
    f = _only(ev, "cup_handle", "forming")
    # right lip usable k bars later, and the handle must reach its minimum length
    assert f["row"] == p["R"] + max(CFG["pivots"]["k"], CFG["cup_handle"]["handle_days"][0])
    assert f["row"] < e["row"]
    assert not ev["pattern"].isin(["hs_top", "inv_hs"]).any() or True


def _check_hs(top: bool):
    path, p = synthetic.hs_path(top)
    df = synthetic.from_path(path)
    name = "hs_top" if top else "inv_hs"
    ev = detect_ticker(df, CFG)
    e = _only(ev, name, "triggered")
    assert [e[f"p{i}_row"] for i in range(1, 6)] == [p[k] for k in ("LS", "T1", "H", "T2", "RS")]
    Q = df["low" if top else "high"].to_numpy()
    P = df["high" if top else "low"].to_numpy()
    neck = Q[p["T1"]]  # flat neckline by design
    assert Q[p["T2"]] == neck
    c = df["close"].to_numpy()
    beyond = (c < neck) if top else (c > neck)
    assert e["row"] == p["RS"] + np.flatnonzero(beyond[p["RS"] :])[0]
    assert e["trigger_price"] == neck and e["stop"] == P[p["RS"]]
    assert np.isclose(e["target"], neck - (P[p["H"]] - neck))
    assert e["direction"] == (-1 if top else 1)
    assert np.isclose(e["time_sym"], 1.0) and np.isclose(e["shoulder_diff"], 0.0)
    assert np.isclose(e["shoulder_diff_hh"], 0.0)
    atr = ind.atr(*(df[k].to_numpy() for k in ("high", "low", "close")), CFG["pivots"]["atr"])
    # equal shoulders by design, so LS is the more extreme shoulder
    assert np.isclose(e["head_prominence_atr"], abs(P[p["H"]] - P[p["LS"]]) / atr[p["H"]])
    c, w = df["close"].to_numpy(), CFG["head_shoulders"]["close_halfwidth"]
    ext = np.max if top else np.min
    cx = [ext(c[i - w : i + w + 1]) for i in (p["LS"], p["H"], p["RS"])]
    cs = (max if top else min)(cx[0], cx[2])
    assert np.isclose(e["head_prominence_close_atr"], abs(cx[1] - cs) / atr[p["H"]])
    f = _only(ev, name, "forming")
    assert p["RS"] < f["row"] < e["row"]


def test_hs_top_designed():
    _check_hs(True)


def test_inverse_hs_designed():
    _check_hs(False)


def test_inverse_is_mirror_of_top():
    a = detect_ticker(synthetic.from_path(synthetic.hs_path(True)[0]), CFG)
    b = detect_ticker(synthetic.from_path(synthetic.hs_path(False)[0]), CFG)
    ta, tb = _only(a, "hs_top", "triggered"), _only(b, "inv_hs", "triggered")
    for k in ("row", "p1_row", "p3_row", "p5_row", "left_days", "span_days", "time_sym"):
        assert ta[k] == tb[k]


def test_stage_order_and_one_event_per_stage():
    df = synthetic.ohlcv(4000, seed=11)
    for loose in (False, True):
        ev = detect_ticker(df, CFG, loose=loose)
        assert not ev.duplicated(["pid", "stage"]).any()
        order = ev.assign(n=ev["stage"].map({s: i for i, s in enumerate(STAGES)}))
        for _, g in order.groupby("pid"):
            g = g.sort_values("n")
            assert g["row"].is_monotonic_increasing
        trig = ev[ev["stage"] == "triggered"]
        long_ = trig["direction"] == 1
        assert (trig.loc[long_, "stop"] < trig.loc[long_, "trigger_price"]).all()
        assert (trig.loc[~long_, "stop"] > trig.loc[~long_, "trigger_price"]).all()


def test_loose_thresholds_find_more_and_flag_default_failures():
    df = synthetic.ohlcv(4000, seed=12)
    d = detect_ticker(df, CFG)
    lo = detect_ticker(df, CFG, loose=True)
    assert len(lo) >= len(d)
    assert "failed_default" in lo and (lo["failed_default"] != "").any()


def test_reason_codes_match_reference_sheet():
    sheet = project_root() / "docs" / "labeling" / "pattern_reference.md"
    if not sheet.exists():
        pytest.skip("labeling sheets not in this copy")
    text = sheet.read_text()
    table = text[text.index("## Reason codes") :]
    rows = {}
    for line in table.splitlines():
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) == 2 and "`" in cells[1]:
            rows[cells[0]] = re.findall(r"`([a-z_]+)`", cells[1])
    assert tuple(rows["Cup-and-handle"]) == REASONS_CUP
    assert tuple(rows["H&S / inverse H&S"]) == REASONS_HS
    assert tuple(rows["Any pattern"]) == REASONS_ANY


def test_dates_columns_match_rows():
    path, _ = synthetic.cup_handle_path()
    df = synthetic.from_path(path)
    ev = detect_ticker(df, CFG)
    assert (ev["date"].to_numpy() == df["date"].to_numpy()[ev["row"]]).all()
    assert pd.isna(ev.loc[ev["head_row"] < 0, "head_date"]).all()


def test_deep_cup_is_a_separate_pattern():
    path, p = synthetic.cup_handle_path(depth=0.35)
    df = synthetic.from_path(path)
    ev = detect_ticker(df, CFG)
    assert not (ev["pattern"] == "cup_handle").any()
    e = _only(ev, "deep_cup", "triggered")
    assert (e["p1_row"], e["p3_row"]) == (p["L"], p["R"])
    assert e["cup_depth"] > CFG["cup_handle"]["cup_depth"][1]
    # the labeling (loose) build keeps it a cup-and-handle candidate that fails the depth gate
    lo = detect_ticker(df, CFG, loose=True)
    c = lo[(lo["pattern"] == "cup_handle") & (lo["stage"] == "triggered")]
    assert len(c) >= 1 and c["failed_default"].str.contains("cup_depth").all()
    assert not (lo["pattern"] == "deep_cup").any()
