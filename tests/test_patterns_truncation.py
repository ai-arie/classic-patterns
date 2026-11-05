"""Truncation invariance of the detectors: events dated <= t computed on data cut at t equal the
full-history events dated <= t (every column, exact). Real-data part needs data/clean."""

import numpy as np
import pandas as pd
import pytest
import synthetic

from classic_patterns.patterns.detect import detect_ticker
from classic_patterns.utils.code_hash import project_root
from classic_patterns.utils.config import load_config

CFG = load_config("detector")
ROOT = project_root()


def _check(df, cuts, vix=None):
    n = 0
    for loose in (False, True):
        full = detect_ticker(df, CFG, vix, loose=loose)
        for i in cuts:
            part = detect_ticker(df.iloc[: i + 1].reset_index(drop=True), CFG, vix, loose=loose)
            exp = full[full["row"] <= i].reset_index(drop=True)
            pd.testing.assert_frame_equal(part, exp, check_exact=True, obj=f"cut {i} loose={loose}")
            n += len(exp)
    return n


def test_truncation_designed_patterns():
    for path in (
        synthetic.cup_handle_path()[0],
        synthetic.cup_handle_path(depth=0.35)[0],  # deep_cup
        synthetic.base_path()[0],
        synthetic.base_path(lows=(94.0, 95.0, 96.0))[0],
        synthetic.triangle_path()[0],
        synthetic.triangle_path(down=True)[0],
        synthetic.hs_path(True)[0],
        synthetic.hs_path(False)[0],
    ):
        df = synthetic.from_path(path)
        n = len(df)
        # every other bar from before the left lip / left shoulder on
        assert _check(df, range(400, n, 2)) > 0


def test_truncation_random_walk():
    df = synthetic.ohlcv(3000, seed=21)
    vix = pd.Series(np.random.default_rng(0).uniform(0, 1, len(df)), index=df["date"])
    rng = np.random.default_rng(3)
    cuts = sorted(rng.choice(np.arange(300, len(df)), 25, replace=False))
    assert _check(df, cuts, vix) > 0


@pytest.mark.real_data
def test_truncation_real_data():
    prices = ROOT / "data" / "clean" / "prices.parquet"
    market = ROOT / "artifacts" / "features" / "market.parquet"
    if not prices.exists() or not market.exists():
        pytest.skip("data/clean or the feature build not available")
    vix = pd.read_parquet(market).set_index("date")["vix_pct252"]
    rng = np.random.default_rng(42)
    tickers = pd.read_parquet(prices, columns=["ticker"])["ticker"].unique()
    pick = list(rng.choice(tickers, 4, replace=False))
    df_all = pd.read_parquet(prices, filters=[("ticker", "in", pick)])
    for _, df in df_all.groupby("ticker"):
        df = df.sort_values("date").reset_index(drop=True)
        cuts = sorted(rng.choice(np.arange(min(300, len(df) - 1), len(df)), 6, replace=False))
        _check(df, cuts, vix)
