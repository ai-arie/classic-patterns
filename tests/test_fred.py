"""FRED download: default User-Agent, retries on network errors, last error raised."""

import pandas as pd
import pytest
import requests

from classic_patterns.data import download as dl

CSV = "observation_date,T10Y2Y\n2026-10-30,0.55\n2026-10-31,.\n2026-11-03,0.57\n"


class _Resp:
    text = CSV

    def raise_for_status(self):
        pass


def _fake_get(fail_times, calls):
    def get(url, **kwargs):
        calls.append(kwargs)
        if len(calls) <= fail_times:
            raise requests.ConnectionError("remote end closed connection")
        return _Resp()

    return get


def test_retries_then_stores(tmp_path, monkeypatch):
    calls, sleeps = [], []
    monkeypatch.setattr(dl.requests, "get", _fake_get(2, calls))
    monkeypatch.setattr(dl.time, "sleep", sleeps.append)
    df = dl.download_fred("T10Y2Y", "https://x/{series}", tmp_path, attempts=3, backoff_s=1)
    assert len(calls) == 3 and sleeps == [1, 2]
    assert all("headers" not in kw for kw in calls)  # FRED stalls custom User-Agents
    assert df["value"].tolist() == [0.55, 0.57]  # "." = missing, dropped
    stored = pd.read_parquet(tmp_path / "T10Y2Y.parquet")
    assert stored["date"].max() == pd.Timestamp("2026-11-03")


def test_raises_after_last_attempt(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(dl.requests, "get", _fake_get(99, calls))
    monkeypatch.setattr(dl.time, "sleep", lambda s: None)
    with pytest.raises(requests.ConnectionError):
        dl.download_fred("T10Y2Y", "https://x/{series}", tmp_path, attempts=3, backoff_s=1)
    assert len(calls) == 3
    assert not (tmp_path / "T10Y2Y.parquet").exists()
