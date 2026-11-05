"""Daily updater with a fake fetch: append, idempotency, re-adjustment, trimming, calendar."""

import numpy as np
import pandas as pd
import pytest

from classic_patterns.data import download as dl
from classic_patterns.data.adjust import build_price_frame
from classic_patterns.data.calendar import last_completed_session
from classic_patterns.data.update import Updater, merge_recent, truncated_history

DATES = pd.bdate_range("2024-01-02", periods=60)
CFG = {
    "batch_size": 2,
    "sleep_seconds": [0, 0],
    "backoff_minutes": [2, 4, 8, 16, 32],
    "max_consecutive_failures": 5,
    "benchmarks": ["SPY"],
    "yfinance": {"period": "max", "interval": "1d", "threads": False},
    "update": {"period": "1mo", "min_overlap_days": 3, "rtol": 1e-4},
}


def yahoo(dates, dividend_on=None, base=100.0) -> pd.DataFrame:
    """Synthetic raw Yahoo frame; a dividend scales Adj Close before its ex-date."""
    close = base + np.arange(len(dates), dtype=float)
    div = np.zeros(len(dates))
    adj = close.copy()
    if dividend_on is not None:
        i = dates.get_loc(pd.Timestamp(dividend_on))
        div[i] = 1.0
        adj[:i] *= 1 - 1.0 / close[i - 1]
    return pd.DataFrame(
        {
            "Open": close,
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
            "Adj Close": adj,
            "Volume": 1e6,
            "Dividends": div,
            "Stock Splits": 0.0,
        },
        index=dates,
    )


def prices(sym, dates, **kw) -> pd.DataFrame:
    return build_price_frame(yahoo(dates, **kw), sym)


def test_merge_appends_new_days_and_is_idempotent():
    full = prices("AAA", DATES)
    stored = full.iloc[:-5].reset_index(drop=True)
    recent = full.iloc[-21:].reset_index(drop=True)
    status, merged = merge_recent(stored, recent, rtol=1e-4, min_overlap=3)
    assert status == "appended"
    pd.testing.assert_frame_equal(merged, full)
    assert merge_recent(merged, recent, rtol=1e-4, min_overlap=3)[0] == "unchanged"


@pytest.mark.parametrize("div_day", [DATES[-10], DATES[-2]])  # inside the overlap / on a new day
def test_merge_redownloads_after_dividend(div_day):
    stored = prices("AAA", DATES[:-5])
    recent = prices("AAA", DATES, dividend_on=div_day).iloc[-21:].reset_index(drop=True)
    assert merge_recent(stored, recent, rtol=1e-4, min_overlap=3) == ("redownload", None)


def test_merge_redownloads_without_enough_overlap():
    stored = prices("AAA", DATES[:30])
    recent = prices("AAA", DATES).iloc[-21:].reset_index(drop=True)  # starts after stored ends
    assert merge_recent(stored, recent, rtol=1e-4, min_overlap=3)[0] == "redownload"


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(dl.time, "sleep", lambda s: None)
    dirs = {"prices": tmp_path / "prices", "benchmarks": tmp_path / "bench"}
    dirs["prices"].mkdir()
    return dl.DownloadState(tmp_path / "state.json"), dirs


def test_updater_end_to_end(env, monkeypatch):
    state, dirs = env
    last = DATES[-2]  # the final DATES row is a session that has not closed yet
    full = {s: prices(s, DATES, base=b) for s, b in (("AAA", 10.0), ("BBB", 50.0), ("NEW", 5.0))}
    full["BBB"] = prices("BBB", DATES, base=50.0, dividend_on=DATES[-3])
    for s in ("AAA", "BBB"):
        prices(s, DATES[:-6], base=10.0 if s == "AAA" else 50.0).to_parquet(
            dirs["prices"] / f"{s}.parquet"
        )
    calls = []

    def fake_fetch(symbols, yf_cfg):
        calls.append((tuple(symbols), yf_cfg["period"]))
        n = None if yf_cfg["period"] == "max" else 21
        return {s: full[s].iloc[-n:].reset_index(drop=True) if n else full[s] for s in symbols}

    monkeypatch.setattr(dl, "fetch", fake_fetch)
    counts = Updater(CFG, state, dirs, last).run(["AAA", "BBB", "NEW"])
    assert counts["appended"] == 1 and counts["redownload"] == 1 and counts["new_listings"] == 1
    for s in ("AAA", "BBB", "NEW"):
        got = pd.read_parquet(dirs["prices"] / f"{s}.parquet")
        pd.testing.assert_frame_equal(got, full[s].iloc[:-1].reset_index(drop=True))
    assert (("NEW",), "max") in calls and (("BBB",), "max") in calls

    calls.clear()
    counts = Updater(CFG, state, dirs, last).run(["AAA", "BBB", "NEW"])
    assert counts == {"unchanged": 3, "new_listings": 0}
    assert all(period == "1mo" for _, period in calls)


def test_truncated_redownload_keeps_stored_history(env, monkeypatch):
    """After a delisting Yahoo returns only the last day(s): the stored history must survive."""
    state, dirs = env
    last = DATES[-2]
    stored = prices("DEL", DATES[:-6])
    stored.to_parquet(dirs["prices"] / "DEL.parquet")
    stub = prices("DEL", DATES[-3:-2])  # one day, no overlap with the stored rows

    monkeypatch.setattr(dl, "fetch", lambda symbols, yf_cfg: {s: stub for s in symbols})
    counts = Updater(CFG, state, dirs, last).run(["DEL"])
    assert counts["redownload"] == 1 and counts["redownload_kept"] == 1
    pd.testing.assert_frame_equal(pd.read_parquet(dirs["prices"] / "DEL.parquet"), stored)


def test_truncated_history_rules():
    stored = pd.Series(pd.bdate_range("2000-01-03", periods=1000))
    assert truncated_history(stored, stored, 31, 0.9) is None
    assert truncated_history(stored, stored.iloc[3:], 31, 0.9) is None  # Yahoo trimmed a few rows
    assert "starts" in truncated_history(stored, stored.iloc[-1:], 31, 0.9)
    assert "rows" in truncated_history(stored, stored.iloc[::2], 31, 0.9)  # gappy
    assert truncated_history(stored, stored.iloc[:0], 31, 0.9) == "is empty"


@pytest.mark.parametrize(
    "now, expected",
    [
        ("2024-07-05 15:00", "2024-07-03"),  # Fri before the close; Jul 4 holiday, Jul 3 half day
        ("2024-07-05 21:30", "2024-07-05"),  # Fri 17:30 ET
        ("2024-07-05 20:30", "2024-07-03"),  # Fri 16:30 ET: inside the settle window
        ("2024-07-07 12:00", "2024-07-05"),  # Sunday
    ],
)
def test_last_completed_session(now, expected):
    got = last_completed_session(pd.Timestamp(now, tz="UTC"), settle_minutes=60)
    assert got == pd.Timestamp(expected)
