import numpy as np
import pandas as pd
import synthetic

from classic_patterns.features import indicators as ind
from classic_patterns.features.eligibility import eligibility_parts, hysteresis, span_mask
from classic_patterns.features.market import macro_features
from classic_patterns.features.weekly import week_end_sessions, weekly_features


def _wilder_loop(x, n):
    out = np.full(len(x), np.nan)
    out[n - 1] = np.mean(x[:n])
    for i in range(n, len(x)):
        out[i] = out[i - 1] + (x[i] - out[i - 1]) / n
    return out


def test_wilder_matches_loop():
    x = np.random.default_rng(0).normal(size=300)
    np.testing.assert_allclose(ind.wilder(x, 14), _wilder_loop(x, 14), rtol=1e-12)


def test_rsi_reference():
    c = synthetic.ohlcv(400, seed=1)["close"].to_numpy()
    d = np.diff(c)
    g = _wilder_loop(np.maximum(d, 0), 14)
    lo = _wilder_loop(np.maximum(-d, 0), 14)
    ref = np.concatenate([[np.nan], 100 - 100 / (1 + g / lo)])
    np.testing.assert_allclose(ind.rsi(c, 14), ref, rtol=1e-10)
    assert np.isnan(ind.rsi(c, 14)[:14]).all() and not np.isnan(ind.rsi(c, 14)[14])


def test_rsi_flat_and_rising():
    assert ind.rsi(np.full(30, 5.0), 14)[-1] == 50.0
    assert ind.rsi(np.arange(1.0, 31.0), 14)[-1] == 100.0


def test_atr_reference():
    df = synthetic.ohlcv(300, seed=2)
    h, lo, c = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
    tr = [h[0] - lo[0]] + [
        max(h[i] - lo[i], abs(h[i] - c[i - 1]), abs(lo[i] - c[i - 1])) for i in range(1, len(c))
    ]
    np.testing.assert_allclose(ind.atr(h, lo, c, 14), _wilder_loop(np.array(tr), 14), rtol=1e-12)


def test_rolling_percentile():
    x = np.array([3.0, 1.0, 2.0, 5.0, 4.0])
    np.testing.assert_allclose(ind.rolling_percentile(x, 3)[2:], [2 / 3, 1.0, 2 / 3])


def test_week_end_holiday_thursday():
    # Good Friday 2024-03-29: the week's last session is Thursday 2024-03-28
    we = week_end_sessions("2024-03-18", "2024-04-12")
    assert list(we.strftime("%Y-%m-%d")) == ["2024-03-22", "2024-03-28", "2024-04-05", "2024-04-12"]


def test_weekly_uses_completed_weeks_only():
    df = synthetic.ohlcv(400, seed=5, drop_rows=0)
    we = week_end_sessions(df["date"].iloc[0], df["date"].iloc[-1])
    cfg = {"rsi": 3, "sma": [2], "volume_window": 2}
    f = weekly_features(df, we, cfg)
    # on non-week-end days the value equals the previous week-end's value
    is_end = df["date"].isin(we).to_numpy()
    col = f["w_dist_sma2_pct"].to_numpy()
    for i in range(30, 400):
        if not is_end[i]:
            j = np.flatnonzero(is_end[:i])[-1]
            assert col[i] == col[j] or (np.isnan(col[i]) and np.isnan(col[j]))
    # week-end value: close vs mean of this and the previous week's close
    j = np.flatnonzero(is_end)[10]
    jp = np.flatnonzero(is_end)[9]
    c = df["close"].to_numpy()
    assert np.isclose(col[j], c[j] / ((c[j] + c[jp]) / 2) - 1)


def test_hysteresis():
    ok = np.array([0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1], dtype=bool)
    np.testing.assert_array_equal(hysteresis(ok, 3).astype(int), [0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1])


def test_eligibility_parts():
    n = 400
    close_raw = np.full(n, 10.0)
    close_raw[350:360] = 4.0  # 10 days under $5 -> off at the 10th day
    dv = np.full(n, 20e6)
    span = np.zeros(n, dtype=bool)
    span[380:385] = True
    cfg = {"min_price": 5.0, "min_history": 300, "hysteresis": True, "off_days": 10}
    p = eligibility_parts(close_raw, dv, span, cfg, 15e6)
    e = p["eligible"]
    assert not e[:299].any() and e[299]
    assert e[350:359].all() and not e[359]
    assert not p["raw"][350:360].any()
    assert e[360] and not e[380:385].any() and e[385]


def test_span_mask_inclusive():
    d = pd.date_range("2020-01-01", periods=10)
    excl = pd.DataFrame(
        {"action": ["exclude_span", "drop_rows"], "start": [d[2], d[0]], "end": [d[4], d[9]]}
    )
    np.testing.assert_array_equal(np.flatnonzero(span_mask(d, excl)), [2, 3, 4])


def test_macro_publication_lag():
    sessions = pd.DatetimeIndex(
        ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"]
    )
    macro = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-06"]),
            "value": [1.0, 2.0, 3.0],
            "series": "T10Y2Y",
        }
    )
    f = macro_features(macro, sessions, {"macro": ["T10Y2Y"], "macro_lag_sessions": 1})
    np.testing.assert_array_equal(f["t10y2y"].to_numpy(), [np.nan, 1.0, 2.0, 2.0, 3.0])
