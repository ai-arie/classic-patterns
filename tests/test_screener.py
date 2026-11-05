"""Which detected setups the screener keeps, and what it shows of them."""

from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from classic_patterns.app.display import with_distances
from classic_patterns.app.screener import (
    TRADE_ESTIMATES,
    blank_forming,
    is_live,
    result_path,
    session_of,
    setups,
)


def test_rerun_of_a_session_keeps_the_first_file(tmp_path):
    day = pd.Timestamp("2026-11-02")
    first = result_path(tmp_path, day)
    assert first.name == "2026-11-02.parquet"
    first.write_text("first")

    rerun = result_path(tmp_path, day, now=datetime(2026, 11, 3, 8, 30, 5))
    assert rerun.name == "2026-11-02_20261103T083005.parquet"
    assert first.read_text() == "first"
    rerun.write_text("rerun")
    next_day = tmp_path / "2026-11-03.parquet"
    next_day.write_text("next")

    # the newest run of the latest session sorts last, for the app and the digest
    assert sorted(tmp_path.glob("*.parquet")) == [first, rerun, next_day]
    assert session_of(first) == session_of(rerun) == "2026-11-02"


def test_is_live():
    # long open, long past target, long past stop, short open, short past stop, short target < 0
    direction = [1, 1, 1, -1, -1, -1]
    close = [100.0, 121.0, 89.0, 100.0, 111.0, 7.83]
    stop = [90.0, 90.0, 90.0, 110.0, 110.0, 14.19]
    target = [120.0, 120.0, 120.0, 80.0, 80.0, -11.02]
    got = is_live(direction, close, stop, target)
    assert got.tolist() == [True, False, False, True, False, False]
    assert got.dtype == np.bool_


def test_setups_are_long_only():
    day = pd.Timestamp("2026-11-02")
    events = pd.DataFrame(
        {
            "ticker": ["LONG", "SHORT", "OTHERDAY"],
            "date": [day, day, day - pd.Timedelta(days=1)],
            "direction": [1, -1, 1],
            "fs_row": [0, 1, 2],
        }
    )
    store = SimpleNamespace(rows=pd.DataFrame({"elig": [1, 1, 1]}))
    assert setups(store, events, day)["ticker"].tolist() == ["LONG"]


def test_forming_has_no_trade_estimates():
    df = pd.DataFrame({"stage": ["triggered", "forming"], **dict.fromkeys(TRADE_ESTIMATES, 0.5)})
    df["student_p_trigger20"] = [np.nan, 0.6]
    out = blank_forming(df)
    assert out.loc[0, TRADE_ESTIMATES].notna().all()
    assert out.loc[1, TRADE_ESTIMATES].isna().all()
    assert out.loc[1, "student_p_trigger20"] == 0.6


def test_distances_use_close_when_triggered_and_trigger_when_forming():
    df = pd.DataFrame(
        {
            "stage": ["triggered", "forming"],
            "close": [100.0, 108.25],
            "trigger": [95.0, 112.39],
            "stop": [90.0, 107.37],
            "target": [120.0, 129.49],
        }
    )
    out = with_distances(df)
    assert out.loc[0, "risk_pct"] == pytest.approx(10.0)
    assert out.loc[0, "target_pct"] == pytest.approx(20.0)
    assert np.isnan(out.loc[0, "to_trigger_pct"])
    assert out.loc[1, "to_trigger_pct"] == pytest.approx(3.82, abs=0.01)
    assert out.loc[1, "risk_pct"] == pytest.approx(4.47, abs=0.01)
    assert out.loc[1, "target_pct"] == pytest.approx(15.21, abs=0.01)
