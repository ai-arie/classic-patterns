"""Per-ticker data validation: integrity checks and suspicious events.

``check_ticker`` never modifies data. It returns a summary row and a list of events
(one row per suspicious date); ``decide`` turns those into cleaning actions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PRICE_COLS = ["open", "high", "low", "close"]
# |log| of common split ratios (2:1 ... 50:1 and their reverse-split inverses). 3:2 and 5:4
# are left out: a 33% / 20% move is far more often a real earnings move than a missed split.
SPLIT_LOG_RATIOS = np.log([2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50])


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """(start, end) index pairs (inclusive) of consecutive True runs."""
    if not mask.any():
        return []
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1, strict=True))


def _median_around(x: np.ndarray, i: int, n: int) -> tuple[float, float]:
    """Median of x over the n rows before i and the n rows from i on (NaN if empty)."""
    before, after = x[max(0, i - n) : i], x[i : i + n]
    med = lambda a: float(np.nanmedian(a)) if np.isfinite(a).any() else np.nan  # noqa: E731
    return med(before), med(after)


def check_ticker(
    df: pd.DataFrame, sessions: pd.DatetimeIndex, cfg: dict, is_benchmark: bool = False
) -> tuple[dict, list[dict]]:
    """Run all checks on one ticker's raw price frame (sorted by date)."""
    t = df["ticker"].iloc[0]
    dates = pd.DatetimeIndex(df["date"])
    n = len(df)
    events: list[dict] = []

    px = df[PRICE_COLS].to_numpy(float)
    o, h, lo, c = px.T
    vol = df["volume"].to_numpy(float)
    dv = (df["close_raw"] * df["volume_raw"]).to_numpy(float)

    # Could this stretch matter? A stock that is "roughly tradable" (loose version of the
    # eligibility filter) within `impact_window` rows after an event can have features or
    # pattern geometry contaminated by it. Benchmarks always matter.
    if is_benchmark:
        relevant = np.ones(n, dtype=bool)
    else:
        dv50 = pd.Series(dv).rolling(50, min_periods=20).mean().to_numpy()
        loose = (df["close_raw"].to_numpy(float) >= cfg["loose_min_price"]) & (
            dv50 >= cfg["loose_min_dollar_volume"]
        )
        w = cfg["impact_window"]
        relevant = pd.Series(loose[::-1]).rolling(w, min_periods=1).max().to_numpy()[::-1] > 0

    def ev(i: int, kind: str, value: float = np.nan, detail: str = "", end: int | None = None):
        events.append(
            {
                "ticker": t,
                "date": dates[i],
                "end": dates[i if end is None else end],
                "kind": kind,
                "value": float(value),
                "detail": detail,
                "relevant": bool(relevant[i]),
            }
        )

    splits = df["splits"].to_numpy(float)
    divs = df["dividends"].to_numpy(float)
    adj = df["adj_factor"].to_numpy(float)

    # --- integrity --------------------------------------------------------------
    nan_rows = np.isnan(px).any(axis=1)
    nonpos_rows = (px <= 0).any(axis=1)
    tol = cfg["ohlc_tolerance"]
    ohlc_bad = (h < np.fmax(o, c) * (1 - tol)) | (lo > np.fmin(o, c) * (1 + tol)) | (h < lo)
    for i in np.flatnonzero(nonpos_rows):
        ev(i, "nonpositive_price", np.nanmin(px[i]))

    check_from = pd.Timestamp(cfg["calendar_check_start"])
    in_scope = dates >= check_from
    off_calendar = in_scope & ~dates.isin(sessions)
    for i in np.flatnonzero(off_calendar):
        ev(i, "off_calendar_date")

    # missing sessions between consecutive rows (calendar checked from check_from on)
    pos = sessions.searchsorted(dates)
    gap_sessions = np.diff(pos) - 1  # missing sessions before row i+1
    gap_sessions[~in_scope[1:]] = 0
    for k in np.flatnonzero(gap_sessions >= cfg["gap_flag_sessions"]):
        i = k + 1
        level = c[i] / c[k] if c[k] > 0 else np.nan
        ev(i, "gap", gap_sessions[k], f"price ratio across gap {level:.3g}")
    missing_sessions = int(gap_sessions[gap_sessions > 0].sum())

    # --- returns / adjustment ---------------------------------------------------
    with np.errstate(divide="ignore", invalid="ignore"):
        logret = np.diff(np.log(c), prepend=np.nan)
        raw_logret = np.diff(np.log(df["close_raw"].to_numpy(float)), prepend=np.nan)
    big = np.abs(logret) > np.log1p(cfg["big_return"])
    split_day = (splits > 0) & (np.abs(np.log(np.where(splits > 0, splits, 1.0))) > 0.05)
    nwin = cfg["volume_window"]

    for i in np.flatnonzero(split_day):
        ratio = splits[i]
        # adjusted series should be continuous across a split; if it jumps by the split
        # ratio instead (and the raw close does not), the split was recorded but not applied
        if abs(logret[i] - np.log(ratio)) < cfg["split_match_tol"] and abs(logret[i]) > 0.1:
            ev(i, "split_not_applied", ratio, f"adj ret {np.expm1(logret[i]):+.3f}")

    reverted: set[int] = set()  # the day a spike reverts is part of the spike event
    for i in np.flatnonzero(big):
        if split_day[i] or i in reverted:
            continue  # split days are handled above (or a genuine move on a split day)
        r = logret[i]
        nxt = logret[i + 1] if i + 1 < n else np.nan
        if np.isfinite(nxt) and abs(r + nxt) < cfg["revert_tol"]:
            # real one-day spikes come with heavy volume; a spike on ordinary volume is
            # most likely a bad print (benchmarks: always treated as real)
            v_med, _ = _median_around(np.where(vol > 0, vol, np.nan), i, 50)
            v_ratio = vol[i] / v_med if np.isfinite(v_med) and v_med > 0 else np.nan
            bad = not is_benchmark and not (v_ratio >= cfg["bad_print_max_volume_ratio"])
            kind = "bad_print" if bad else "spike_revert"
            ev(i, kind, np.expm1(r), f"next {np.expm1(nxt):+.3f}, volume x{v_ratio:.3g}")
            reverted.add(i + 1)
            continue
        if not is_benchmark:
            dv_b, dv_a = _median_around(dv, i, nwin)
            v_b, v_a = _median_around(np.where(vol > 0, vol, np.nan), i, nwin)
            # a missed split moves the price by the ratio and the share volume by its
            # inverse, leaving dollar volume roughly unchanged
            if all(np.isfinite([dv_b, dv_a, v_b, v_a])) and min(dv_b, dv_a, v_b, v_a) > 0:
                dv_jump = np.log(dv_a / dv_b)
                v_jump = np.log(v_a / v_b)
                near_ratio = np.min(np.abs(abs(r) - SPLIT_LOG_RATIOS)) < cfg["split_match_tol"]
                if near_ratio and abs(dv_jump) < np.log(1.5) and abs(v_jump + r) < np.log(1.25):
                    ev(i, "suspect_missing_split", np.expm1(r), f"vol x{np.exp(v_jump):.3g}")
                    continue
        ev(i, "big_move", np.expm1(r))

    # Large dividends: real ones (spin-offs, specials) show up as an ex-date price drop of
    # about the dividend. Otherwise the record is corrupt and it mis-scales all earlier
    # adjusted prices (by a constant; negative or zero if the dividend exceeds the price).
    y_close, y_open = df["y_close"].to_numpy(float), df["y_open"].to_numpy(float)
    prev_close = np.concatenate([[np.nan], y_close[:-1]])
    for i in np.flatnonzero(divs > cfg["large_dividend_ratio"] * prev_close):
        drop = (prev_close[i] - y_open[i]) / divs[i]
        adj_before = adj[i - 1]
        detail = f"prev close {prev_close[i]:.4g}, ex-date drop / dividend {drop:.2f}"
        if not adj_before > cfg["min_adj_factor"]:
            ev(i, "bad_dividend_nonpositive", divs[i], detail)
        elif not drop >= cfg["dividend_min_drop_ratio"]:
            ev(i, "bad_dividend", divs[i], detail)
        else:
            ev(i, "large_dividend", divs[i], detail)

    with np.errstate(divide="ignore", invalid="ignore"):
        adj_step = np.diff(np.log(adj), prepend=np.nan)
    # dividend within +-2 rows
    has_div = pd.Series(divs > 0).rolling(5, center=True, min_periods=1).max().to_numpy() > 0
    for i in np.flatnonzero((np.abs(adj_step) > cfg["adj_jump_tol"]) & ~has_div):
        ev(i, "adj_jump_unexplained", np.expm1(adj_step[i]))

    # --- volume / staleness -----------------------------------------------------
    zero_vol = (vol == 0) | np.isnan(vol)
    lead_zero = int(np.argmax(~zero_vol)) if (~zero_vol).any() else n
    flat = np.concatenate([[False], np.diff(c) == 0])
    stale_runs = [] if is_benchmark else _runs(flat & zero_vol)
    stale_runs = [(a, b) for a, b in stale_runs if b - a + 1 >= cfg["stale_run_days"]]
    for a, b in stale_runs:
        ev(a, "stale_run", b - a + 1, end=b)

    summary = {
        "ticker": t,
        "is_benchmark": is_benchmark,
        "rows": n,
        "first": dates[0],
        "last": dates[-1],
        "duplicate_dates": int(dates.duplicated().sum()),
        "nan_price_rows": int(nan_rows.sum()),
        "nonpositive_rows": int(nonpos_rows.sum()),
        "ohlc_bad_rows": int(ohlc_bad.sum()),
        "off_calendar_rows": int(off_calendar.sum()),
        "missing_sessions": missing_sessions,
        "max_gap_sessions": int(gap_sessions.max()) if n > 1 else 0,
        "zero_volume_share": float(zero_vol.mean()) if not is_benchmark else np.nan,
        "leading_zero_volume_rows": lead_zero if not is_benchmark else 0,
        "stale_run_days": int(sum(b - a + 1 for a, b in stale_runs)),
        "split_days": int(split_day.sum()),
        "fractional_split_days": int(
            (split_day & (np.abs(splits - np.round(splits)) > 1e-6) & (splits > 1)).sum()
        ),
        "dividend_days": int((divs > 0).sum()),
        "max_abs_raw_logret": float(np.nanmax(np.abs(raw_logret))) if n > 1 else np.nan,
    }
    for kind in (
        "split_not_applied",
        "spike_revert",
        "suspect_missing_split",
        "big_move",
        "adj_jump_unexplained",
        "bad_print",
        "bad_dividend",
        "bad_dividend_nonpositive",
        "large_dividend",
        "stale_run",
        "gap",
    ):
        summary[f"n_{kind}"] = sum(e["kind"] == kind for e in events)
    return summary, events


# ---------------------------------------------------------------------------
# cleaning decision
# ---------------------------------------------------------------------------

ACTION_COLS = ["ticker", "action", "reason", "start", "end", "event_date", "detail"]


def decide(
    df: pd.DataFrame, events: list[dict], cfg: dict, is_benchmark: bool = False
) -> list[dict]:
    """Cleaning actions for one ticker.

    Actions: ``drop_ticker``; ``truncate_before`` (keep rows on/after ``start``);
    ``drop_rows`` (remove rows in [start, end]); ``exclude_span`` (rows stay in the data but
    are never eligible as trigger dates / samples). ``event_date`` is the date of the artifact
    for spans created by an event (NaT otherwise).
    """
    t = df["ticker"].iloc[0]
    dates = pd.DatetimeIndex(df["date"])
    out: list[dict] = []

    def act(action, reason, start, end=None, detail="", event_date=None):
        out.append(
            {
                "ticker": t,
                "action": action,
                "reason": reason,
                "start": pd.Timestamp(start),
                "end": pd.Timestamp(start if end is None else end),
                "event_date": pd.Timestamp(event_date) if event_date is not None else pd.NaT,
                "detail": detail,
            }
        )

    # 1. truncation: corrupt dividend that makes earlier adjusted prices <= 0, and any
    #    remaining non-positive / NaN prices (keep only the rows after the last bad one)
    px = df[PRICE_COLS].to_numpy(float)
    bad_px = np.isnan(px).any(axis=1) | (px <= 0).any(axis=1)
    cut = dates[0]
    for e in events:
        if e["kind"] == "bad_dividend_nonpositive" and e["date"] > cut:
            cut, why = e["date"], "bad_dividend_nonpositive"
    if bad_px.any():
        last_bad = np.flatnonzero(bad_px)[-1]
        if last_bad + 1 >= len(dates):
            act("drop_ticker", "nonpositive_or_nan_prices", dates[0], dates[-1])
            return out
        if dates[last_bad + 1] > cut:
            cut, why = dates[last_bad + 1], "nonpositive_or_nan_prices"
    if cut > dates[0]:
        act("truncate_before", why, cut, detail=f"{int((dates < cut).sum())} rows removed")
    keep = dates >= cut

    # 2. rows to delete: off-calendar dates, zero-volume placeholders before trading starts
    for e in events:
        if e["kind"] == "off_calendar_date" and e["date"] >= cut:
            act("drop_rows", "off_calendar_date", e["date"])
            keep &= dates != e["date"]
    if not is_benchmark:
        vol = df["volume"].to_numpy(float)
        idx = np.flatnonzero(keep)
        traded = idx[vol[idx] > 0]
        if len(traded) == 0:
            act("drop_ticker", "no_volume", dates[0], dates[-1])
            return out
        lead = idx[idx < traded[0]]
        if len(lead):
            act(
                "drop_rows",
                "leading_zero_volume",
                dates[lead[0]],
                dates[lead[-1]],
                f"{len(lead)} rows",
            )
            keep[lead] = False

    n_keep = int(keep.sum())
    if n_keep < cfg["min_rows"]:
        act("drop_ticker", "short_history", dates[0], dates[-1], f"{n_keep} rows after cleaning")
        return out

    # 3. exclusion spans (rows kept), positions counted in the cleaned rows
    kept_dates = dates[keep]
    for e in events:
        if e["date"] < kept_dates[0] or not keep[dates.get_loc(e["date"])]:
            continue
        if e["kind"] in cfg["span_kinds"]:
            i = kept_dates.get_loc(e["date"])
            a = max(0, i - cfg["span_rows_before"])
            b = min(len(kept_dates) - 1, i + cfg["span_rows_after"])
            act(
                "exclude_span",
                e["kind"],
                kept_dates[a],
                kept_dates[b],
                f"event {e['date'].date()}: {e['detail']}",
                event_date=e["date"],
            )
        elif e["kind"] == "stale_run" and cfg["stale_runs"] == "exclude":
            act(
                "exclude_span",
                "stale_run",
                e["date"],
                e["end"],
                f"{int(e['value'])} rows",
                event_date=e["date"],
            )
    return out
