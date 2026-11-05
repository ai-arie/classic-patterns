"""Validation checks, cleaning decisions and exclusion application on synthetic data."""

import numpy as np
import pandas as pd
import pytest

from classic_patterns.data.adjust import build_price_frame
from classic_patterns.data.calendar import nyse_sessions
from classic_patterns.data.store import apply_exclusions
from classic_patterns.data.validate import ACTION_COLS, check_ticker, decide
from classic_patterns.utils.config import load_config

CFG = load_config("validate")
CHECKS, CLEAN = CFG["checks"], CFG["cleaning"]
SESSIONS = nyse_sessions("2010-01-01", "2016-12-31")
N = 1000
DATES = SESSIONS[:N]


def yahoo(close, volume=1e6, dividends=0.0, splits=0.0, adj=None) -> pd.DataFrame:
    close = np.asarray(close, float)
    return pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Adj Close": close if adj is None else adj,
            "Volume": volume,
            "Dividends": dividends,
            "Stock Splits": splits,
        },
        index=DATES[: len(close)],
    )


def base_close(seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 50 * np.exp(np.cumsum(rng.normal(0, 0.01, N)))


def run(raw: pd.DataFrame, is_benchmark=False):
    df = build_price_frame(raw, "TEST")
    summary, events = check_ticker(df, SESSIONS, CHECKS, is_benchmark)
    return df, summary, events, decide(df, events, CLEAN, is_benchmark)


def kinds(events):
    return [e["kind"] for e in events]


def test_clean_series_has_no_events_or_actions():
    _, summary, events, actions = run(yahoo(base_close()))
    assert events == [] and actions == []
    assert summary["rows"] == N and summary["missing_sessions"] == 0


def test_split_recorded_but_not_applied_gives_exclusion_span():
    close = base_close()
    close[500:] *= 4  # split-adjusted close jumps by the ratio: Yahoo did not adjust history
    splits = np.zeros(N)
    splits[500] = 4.0
    _, _, events, actions = run(yahoo(close, splits=splits))
    assert kinds(events) == ["split_not_applied"]
    (a,) = actions
    assert a["action"] == "exclude_span" and a["reason"] == "split_not_applied"
    assert a["start"] == DATES[500 - CLEAN["span_rows_before"]]
    assert a["end"] == DATES[min(N - 1, 500 + CLEAN["span_rows_after"])]  # clipped at the end


def test_corrupt_dividend_with_negative_adjustment_truncates():
    close = base_close()
    div = np.zeros(N)
    div[400] = 10 * close[399]  # dividend 10x the price
    adj = close.copy()
    adj[:400] *= 1 - div[400] / close[399]  # negative factor before the ex-date
    df, _, events, actions = run(yahoo(close, dividends=div, adj=adj))
    assert "bad_dividend_nonpositive" in kinds(events)
    trunc = [a for a in actions if a["action"] == "truncate_before"]
    assert trunc and trunc[0]["start"] == DATES[400]
    cleaned = apply_exclusions(df, pd.DataFrame(actions, columns=ACTION_COLS))
    assert cleaned["date"].iloc[0] == DATES[400]
    assert (cleaned[["open", "high", "low", "close"]] > 0).all().all()


def test_real_spinoff_dividend_is_only_flagged():
    close = base_close()
    close[300:] -= 0.6 * close[299]  # price drops by the distributed value
    div = np.zeros(N)
    div[300] = 0.6 * close[299]
    _, _, events, actions = run(yahoo(close, dividends=div))
    assert "large_dividend" in kinds(events) and actions == []


def test_spike_on_normal_volume_is_bad_print_but_heavy_volume_spike_is_real():
    close = base_close()
    close[600] *= 2
    vol = np.full(N, 1e6)
    _, _, events, actions = run(yahoo(close, volume=vol))
    assert kinds(events) == ["bad_print"]
    assert actions[0]["reason"] == "bad_print"
    vol[600] = 2e7
    _, _, events, actions = run(yahoo(close, volume=vol))
    assert kinds(events) == ["spike_revert"] and actions == []


def test_leading_zero_volume_and_stale_run():
    close = base_close()
    close[700:720] = close[699]
    vol = np.full(N, 1e6)
    vol[:5] = 0
    vol[700:720] = 0
    df, _, events, actions = run(yahoo(close, volume=vol))
    by = {(a["action"], a["reason"]): a for a in actions}
    lead = by[("drop_rows", "leading_zero_volume")]
    assert (lead["start"], lead["end"]) == (DATES[0], DATES[4])
    stale = by[("exclude_span", "stale_run")]
    assert stale["start"] == DATES[700] and stale["end"] == DATES[719]
    cleaned = apply_exclusions(df, pd.DataFrame(actions, columns=ACTION_COLS))
    assert len(cleaned) == N - 5  # the stale run stays in the data


@pytest.mark.parametrize("rows, dropped", [(299, True), (300, False)])
def test_short_history_drops_ticker(rows, dropped):
    df, _, _, actions = run(yahoo(base_close()[:rows]))
    assert any(a["action"] == "drop_ticker" for a in actions) == dropped
    assert (apply_exclusions(df, pd.DataFrame(actions, columns=ACTION_COLS)) is None) == dropped


def test_benchmark_spike_is_never_a_bad_print():
    close = base_close()
    close[600] *= 2
    _, _, events, actions = run(yahoo(close, volume=0.0), is_benchmark=True)
    assert kinds(events) == ["spike_revert"] and actions == []


def test_context_guards_cut_at_level_shift_and_mask_bad_print():
    from classic_patterns.data.store import context_guards

    close = base_close()
    close[500:] *= 4
    close[800] *= 2  # bad print on normal volume
    splits = np.zeros(N)
    splits[500] = 4.0
    df, _, _, actions = run(yahoo(close, splits=splits))
    excl = pd.DataFrame(actions, columns=ACTION_COLS)
    assert set(excl["reason"]) == {"split_not_applied", "bad_print"}
    assert (excl["event_date"] == pd.Series([DATES[500], DATES[800]])).all()
    floor, bad = context_guards(df["date"], excl)
    assert (floor[:500] == 0).all() and (floor[500:] == 500).all()
    assert np.flatnonzero(bad).tolist() == [800]
