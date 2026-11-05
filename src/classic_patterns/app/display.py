"""What the app, the email digest and the console show of a screener result (long setups)."""

from __future__ import annotations

import pandas as pd

TARGET_NOTE = (
    "Targets are the patterns' textbook measured moves and are usually far away. Setting "
    "take-profit levels was not part of this project, so use your own judgement, especially with "
    "the targets."
)


def with_distances(df: pd.DataFrame) -> pd.DataFrame:
    """Adds, in percent: ``to_trigger_pct`` (forming only), ``risk_pct`` and ``target_pct``.
    Triggered setups are measured from the close, forming setups from their trigger level, the
    price at which the trade would start."""
    out = df.copy()
    forming = out["stage"] == "forming"
    entry = out["trigger"].where(forming, out["close"])
    out["to_trigger_pct"] = ((out["trigger"] / out["close"] - 1) * 100).where(forming)
    out["risk_pct"] = (entry - out["stop"]) / entry * 100
    out["target_pct"] = (out["target"] / entry - 1) * 100
    return out
