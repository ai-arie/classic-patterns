"""NYSE trading calendar helpers (dates are tz-naive, normalized)."""

from __future__ import annotations

import pandas as pd
import pandas_market_calendars as mcal


def nyse_sessions(start, end) -> pd.DatetimeIndex:
    """NYSE sessions in [start, end] as tz-naive datetime64[ns] (includes pre-1952 Saturdays)."""
    days = mcal.get_calendar("NYSE").valid_days(start_date=start, end_date=end)
    return pd.DatetimeIndex(days.tz_localize(None).normalize()).as_unit("ns")


def last_completed_session(
    now: pd.Timestamp | None = None, settle_minutes: int = 60
) -> pd.Timestamp:
    """Most recent NYSE session whose close is at least ``settle_minutes`` before ``now``."""
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if now.tz is None:
        now = now.tz_localize("UTC")
    sched = mcal.get_calendar("NYSE").schedule(
        start_date=(now - pd.Timedelta(days=14)).date(), end_date=now.date()
    )
    done = sched[sched["market_close"] + pd.Timedelta(minutes=settle_minutes) <= now]
    return pd.Timestamp(done.index[-1]).normalize().as_unit("ns")
