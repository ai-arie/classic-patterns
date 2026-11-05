import numpy as np
import pandas as pd
import pytest

from classic_patterns.data.adjust import OUTPUT_COLS, build_price_frame, future_split_factor


def test_future_split_factor_excludes_split_day():
    s = pd.Series([0, 0, 2.0, 0, 0.1, 0])
    f = future_split_factor(s)
    assert f.tolist() == pytest.approx([0.2, 0.2, 0.1, 0.1, 1.0, 1.0])


def make_yahoo_frame() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Synthetic stock: 2:1 split on day 3, 1:10 reverse split on day 6, dividend on day 5.

    Returns (yahoo frame, true traded close, true share volume).
    """
    dates = pd.date_range("2020-01-02", periods=8, freq="B", tz="America/New_York")
    true_close = pd.Series([100, 102, 51, 52, 53, 5.0 * 10 + 1, 51.5, 52], index=dates, dtype=float)
    true_vol = pd.Series([1000, 1100, 2400, 2000, 2100, 250, 260, 240], index=dates, dtype=float)
    splits = pd.Series([0, 0, 2.0, 0, 0, 0.1, 0, 0], index=dates)

    # Yahoo divides prices / multiplies volume by all later split ratios.
    later = future_split_factor(splits.reset_index(drop=True)).to_numpy()
    y_close = true_close / later
    y_vol = true_vol * later
    divs = pd.Series([0, 0, 0, 0, 1.0, 0, 0, 0], index=dates)
    # dividend-adjustment factor: 1 - div/prev_close applied to dates before the ex-date
    adj = np.ones(len(dates))
    adj[:4] = 1 - 1.0 / y_close.iloc[3]
    yahoo = pd.DataFrame(
        {
            "Open": y_close * 0.99,
            "High": y_close * 1.02,
            "Low": y_close * 0.97,
            "Close": y_close,
            "Adj Close": y_close * adj,
            "Volume": y_vol,
            "Dividends": divs,
            "Stock Splits": splits,
        }
    )
    return yahoo, true_close, true_vol


def test_build_price_frame_raw_and_adjusted():
    yahoo, true_close, true_vol = make_yahoo_frame()
    out = build_price_frame(yahoo, "TEST")

    assert list(out.columns) == OUTPUT_COLS
    assert out["date"].dtype == "datetime64[ns]"
    assert out["date"].iloc[0] == pd.Timestamp("2020-01-02")
    np.testing.assert_allclose(out["close_raw"], true_close.to_numpy())
    np.testing.assert_allclose(out["volume_raw"], true_vol.to_numpy())
    # dollar volume is invariant to splits
    np.testing.assert_allclose(
        out["close_raw"] * out["volume_raw"], out["y_close"] * out["y_volume"]
    )
    # adjusted OHLC keeps intraday geometry
    np.testing.assert_allclose(out["high"] / out["close"], 1.02)
    np.testing.assert_allclose(out["low"] / out["close"], 0.97)


def test_build_price_frame_drops_empty_rows_and_fills_actions():
    yahoo, _, _ = make_yahoo_frame()
    yahoo = yahoo.drop(columns=["Dividends", "Stock Splits"])
    yahoo.iloc[0, :4] = np.nan
    out = build_price_frame(yahoo, "TEST")
    assert len(out) == len(yahoo) - 1
    assert (out["splits"] == 0).all() and (out["dividends"] == 0).all()
    np.testing.assert_allclose(out["close_raw"], out["y_close"])
