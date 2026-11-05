"""Checks on the 20-ticker sample (``scripts/download_prices.py --test``). Skipped if absent."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from classic_patterns.utils.code_hash import project_root
from classic_patterns.utils.config import load_config

RAW = project_root() / "data" / "raw"
TEST_TICKERS = set(load_config("download")["test_tickers"])
pytestmark = pytest.mark.sample_data


def load(sym: str) -> pd.DataFrame:
    for sub in ("prices", "benchmarks"):
        p = RAW / sub / f"{sym}.parquet"
        if p.exists():
            return pd.read_parquet(p).set_index("date")
    pytest.skip(f"sample data for {sym} not downloaded")


def sample_files() -> list[Path]:
    files = sorted(RAW.glob("prices/*.parquet")) + sorted(RAW.glob("benchmarks/*.parquet"))
    files = [f for f in files if f.stem in TEST_TICKERS]  # the full universe is checked in A3
    if not files:
        pytest.skip("sample data not downloaded")
    return files


def around(df: pd.DataFrame, day: str) -> tuple[pd.Series, pd.Series]:
    i = df.index.get_loc(pd.Timestamp(day))
    return df.iloc[i - 1], df.iloc[i]


# (symbol, split date, ratio)
FORWARD = [
    ("AAPL", "2020-08-31", 4),
    ("NVDA", "2024-06-10", 10),
    ("TSLA", "2020-08-31", 5),
    ("GOOGL", "2022-07-18", 20),
    ("AMZN", "2022-06-06", 20),
]
REVERSE = [("GE", "2021-08-02", 1 / 8), ("C", "2011-05-09", 1 / 10), ("SIRI", "2024-09-10", 1 / 10)]


@pytest.mark.parametrize("sym, day, ratio", FORWARD + REVERSE)
def test_split_recorded_and_raw_price_jumps(sym, day, ratio):
    df = load(sym)
    prev, cur = around(df, day)
    assert cur["splits"] == pytest.approx(ratio, rel=1e-3)
    raw_jump = prev["close_raw"] / cur["close_raw"]
    assert raw_jump == pytest.approx(ratio, rel=0.15)
    # adjusted series has no jump
    assert abs(np.log(cur["close"] / prev["close"])) < 0.15


@pytest.mark.parametrize("sym, day, ratio", FORWARD + REVERSE)
def test_split_volume_raw_jumps_dollar_volume_continuous(sym, day, ratio):
    df = load(sym)
    prev, cur = around(df, day)
    raw_vol_ratio = (cur["volume_raw"] / prev["volume_raw"]) / (cur["volume"] / prev["volume"])
    assert raw_vol_ratio == pytest.approx(ratio, rel=1e-6)
    # median of 5 days each side: single days can be outliers (C has one on 2011-05-06)
    dv = df["close_raw"] * df["volume_raw"]
    i = df.index.get_loc(pd.Timestamp(day))
    assert 0.2 < dv.iloc[i : i + 5].median() / dv.iloc[i - 5 : i].median() < 5


@pytest.mark.parametrize(
    "sym, day, close",
    [
        ("AAPL", "2020-08-28", 499.23),
        ("TSLA", "2020-08-28", 2213.40),
        ("NVDA", "2024-06-07", 1208.88),
    ],
)
def test_known_pre_split_close(sym, day, close):
    assert load(sym).loc[day, "close_raw"] == pytest.approx(close, rel=0.005)


def test_all_split_dates_have_no_adjusted_jump():
    for f in sample_files():
        df = pd.read_parquet(f).set_index("date")
        ret = np.log(df["close"]).diff()
        split_days = df.index[(df["splits"] > 0) & ((df["splits"] - 1).abs() > 0.05)]
        big = ret.loc[split_days].abs() > 0.25
        assert not big.any(), f"{f.stem}: adjusted jump on split days {list(split_days[big])}"


def test_frame_invariants():
    for f in sample_files():
        df = pd.read_parquet(f)
        assert df["date"].dtype == "datetime64[ns]", f.stem
        assert df["date"].is_monotonic_increasing and df["date"].is_unique, f.stem
        px = df[["open", "high", "low", "close"]]
        assert px.notna().all().all() and (px > 0).all().all(), f.stem
        tol = 1e-6
        assert (df["high"] >= np.maximum(df["open"], df["close"]) * (1 - tol)).all(), f.stem
        assert (df["low"] <= np.minimum(df["open"], df["close"]) * (1 + tol)).all(), f.stem
        np.testing.assert_allclose(
            df["close_raw"] * df["volume_raw"],
            df["y_close"] * df["y_volume"],
            rtol=1e-9,
            err_msg=f.stem,
        )
