"""Truncation invariance: features for date t computed on data cut at t == computed on the full
history. Synthetic tests always run; the real-data ones need data/clean."""

import numpy as np
import pandas as pd
import pytest
import synthetic

from classic_patterns.features.build import ticker_features
from classic_patterns.features.eligibility import eligibility_parts, span_mask
from classic_patterns.features.market import market_context
from classic_patterns.features.stock import dollar_volume
from classic_patterns.features.weekly import week_end_sessions
from classic_patterns.utils.code_hash import project_root
from classic_patterns.utils.config import load_config

CFG = load_config("features")
CLEAN = project_root() / "data" / "clean"


def _features(df, bench_close):
    we = week_end_sessions(df["date"].iloc[0], df["date"].iloc[-1])
    return ticker_features(df, bench_close, we, CFG)


def _check_stock(df, bench_close, cuts, require_warm=False):
    full = _features(df, bench_close)
    if require_warm:
        assert full.iloc[-1].notna().all(), "some feature never warms up"
    for i in cuts:
        t = df["date"].iloc[i]
        part = _features(df.iloc[: i + 1].reset_index(drop=True), bench_close[:t])
        np.testing.assert_array_equal(
            part.iloc[i].to_numpy(),
            full.iloc[i].to_numpy(),
            err_msg=f"row {i} ({df['date'].iloc[i].date()})",
        )


def _check_eligibility(df, excl, cuts):
    ecfg = CFG["eligibility"]
    w = CFG["stock"]["dollar_volume_window"]

    def elig(d):
        dv = dollar_volume(d, w)
        p = eligibility_parts(
            d["close_raw"], dv, span_mask(d["date"], excl), ecfg, ecfg["min_dollar_volume"]
        )
        return p["eligible"]

    full = elig(df)
    for i in cuts:
        assert elig(df.iloc[: i + 1].reset_index(drop=True))[i] == full[i]


def _check_market(bench, macro, cuts_dates):
    first = bench["date"].min()
    end = bench["date"].max()
    from classic_patterns.data.calendar import nyse_sessions

    full = market_context(
        bench, macro, nyse_sessions(first, end), CFG["market"], CFG["stock"]["trend"]
    )
    for t in cuts_dates:
        part = market_context(
            bench[bench["date"] <= t],
            macro[macro["date"] <= t],
            nyse_sessions(first, t),
            CFG["market"],
            CFG["stock"]["trend"],
        )
        np.testing.assert_array_equal(
            part.loc[t].to_numpy(), full.loc[t].to_numpy(), err_msg=str(t.date())
        )


def _cuts(n, k, rng, lo=0):
    return sorted(rng.choice(np.arange(lo, n), size=min(k, n - lo), replace=False))


def test_stock_features_truncation_synthetic():
    df = synthetic.ohlcv(1500, seed=3)
    bench = synthetic.benchmarks()
    gspc = bench[bench["ticker"] == "^GSPC"].set_index("date")["close"]
    rng = np.random.default_rng(0)
    # all warm-up boundaries plus random rows
    cuts = sorted({*range(0, 30), 49, 50, 51, 199, 200, 251, 252, 253, *_cuts(len(df), 40, rng)})
    _check_stock(df, gspc, cuts, require_warm=True)


def test_eligibility_truncation_synthetic():
    df = synthetic.ohlcv(1500, seed=4)
    d = df["date"]
    excl = pd.DataFrame(
        {
            "ticker": "SYN",
            "action": ["exclude_span"],
            "reason": ["bad_print"],
            "start": [d.iloc[700]],
            "end": [d.iloc[760]],
            "event_date": [d.iloc[720]],
            "detail": [""],
        }
    )
    rng = np.random.default_rng(1)
    _check_eligibility(df, excl, _cuts(len(df), 60, rng))


def test_market_context_truncation_synthetic():
    bench = synthetic.benchmarks()
    macro = synthetic.macro()
    dates = np.sort(bench.loc[bench["ticker"] == "^GSPC", "date"].unique())
    rng = np.random.default_rng(2)
    cuts = [pd.Timestamp(dates[i]) for i in _cuts(len(dates), 25, rng, lo=260)]
    _check_market(bench, macro, cuts)


@pytest.mark.real_data
def test_truncation_real_data():
    if not (CLEAN / "prices.parquet").exists():
        pytest.skip("data/clean not available")
    rng = np.random.default_rng(42)
    tickers = pd.read_parquet(CLEAN / "prices.parquet", columns=["ticker"])["ticker"].unique()
    pick = list(rng.choice(tickers, 8, replace=False))
    prices = pd.read_parquet(CLEAN / "prices.parquet", filters=[("ticker", "in", pick)])
    bench = pd.read_parquet(CLEAN / "benchmarks.parquet")
    macro = pd.read_parquet(CLEAN / "macro.parquet")
    excl = pd.read_parquet(project_root() / "data" / "meta" / "exclusions.parquet")
    gspc = bench[bench["ticker"] == "^GSPC"].set_index("date")["close"].sort_index()
    for t, df in prices.groupby("ticker"):
        df = df.sort_values("date").reset_index(drop=True)
        cuts = _cuts(len(df), 6, rng, lo=260)
        _check_stock(df, gspc, cuts)
        _check_eligibility(df, excl[excl["ticker"] == t], cuts)
    sess = np.sort(bench.loc[(bench["ticker"] == "^GSPC") & (bench["date"] >= "1996"), "date"])
    _check_market(bench, macro, [pd.Timestamp(sess[i]) for i in _cuts(len(sess), 6, rng)])
