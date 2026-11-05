"""Downloader control flow with a fake fetch: resume, empties, backoff, clean stop."""

import pandas as pd
import pytest

from classic_patterns.data import download as dl

CFG = {
    "batch_size": 2,
    "sleep_seconds": [0, 0],
    "backoff_minutes": [2, 4, 8, 16, 32],
    "max_consecutive_failures": 5,
    "benchmarks": ["SPY"],
    "yfinance": {"period": "max", "interval": "1d", "threads": False},
}


def frame(sym: str) -> pd.DataFrame:
    return pd.DataFrame(
        {"ticker": sym, "date": pd.to_datetime(["2024-01-02", "2024-01-03"]), "close": [1.0, 2.0]}
    )


@pytest.fixture
def env(tmp_path, monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(dl.time, "sleep", lambda s: sleeps.append(s))
    dirs = {"prices": tmp_path / "prices", "benchmarks": tmp_path / "bench"}
    state = dl.DownloadState(tmp_path / "state.json")
    return state, dirs, sleeps


def test_resume_and_empty_symbols(env, monkeypatch):
    state, dirs, _ = env
    state.mark_done("AAA", frame("AAA"))
    calls = []

    def fake_fetch(symbols, yf_cfg):
        calls.append(list(symbols))
        return {s: None if s == "DEAD" else frame(s) for s in symbols}

    monkeypatch.setattr(dl, "fetch", fake_fetch)
    dl.Downloader(CFG, state, dirs).run(["AAA", "BBB", "DEAD", "SPY"])

    assert ["AAA"] not in calls and all("AAA" not in c for c in calls)
    assert ["DEAD"] in calls  # individual retry
    assert set(state.done) == {"AAA", "BBB", "SPY"}
    assert set(state.empty) == {"DEAD"} and not state.failed
    assert (dirs["prices"] / "BBB.parquet").exists()
    assert (dirs["benchmarks"] / "SPY.parquet").exists()
    reloaded = dl.DownloadState.load(state.path)
    assert set(reloaded.done) == {"AAA", "BBB", "SPY"}


def test_backoff_then_success(env, monkeypatch):
    state, dirs, sleeps = env
    n = {"calls": 0}

    def fake_fetch(symbols, yf_cfg):
        n["calls"] += 1
        if n["calls"] <= 2:
            raise dl.RateLimited("429")
        return {s: frame(s) for s in symbols}

    monkeypatch.setattr(dl, "fetch", fake_fetch)
    dl.Downloader(CFG, state, dirs).run(["BBB"])
    assert sleeps[:2] == [120, 240]
    assert "BBB" in state.done


def test_whole_batch_empty_with_failed_probe_backs_off(env, monkeypatch):
    state, dirs, sleeps = env
    n = {"calls": 0}

    def fake_fetch(symbols, yf_cfg):
        n["calls"] += 1
        if n["calls"] == 1 or symbols == [dl.PROBE_SYMBOL] and n["calls"] == 2:
            return {s: None for s in symbols}
        return {s: frame(s) for s in symbols}

    monkeypatch.setattr(dl, "fetch", fake_fetch)
    dl.Downloader(CFG, state, dirs).run(["BBB", "CCC"])
    assert sleeps[0] == 120
    assert set(state.done) == {"BBB", "CCC"}


def test_stops_cleanly_after_max_failures(env, monkeypatch):
    state, dirs, sleeps = env

    def fake_fetch(symbols, yf_cfg):
        raise dl.RateLimited("429")

    monkeypatch.setattr(dl, "fetch", fake_fetch)
    with pytest.raises(dl.StopDownload):
        dl.Downloader(CFG, state, dirs).run(["BBB", "CCC"])
    assert sleeps == [120, 240, 480, 960]  # 4 backoffs, 5th failure stops
    assert state.path.exists()
