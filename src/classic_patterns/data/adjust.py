"""Build adjusted / unadjusted price columns from a raw Yahoo frame.

Yahoo semantics (``auto_adjust=False``): ``Close`` is split-adjusted but not
dividend-adjusted, ``Adj Close`` is split- and dividend-adjusted, ``Volume`` is
split-adjusted. ``Stock Splits`` holds the split ratio on the effective date
(4.0 for 4:1, 0.125 for a 1:8 reverse split) and 0 otherwise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

YAHOO_COLS = {
    "Open": "y_open",
    "High": "y_high",
    "Low": "y_low",
    "Close": "y_close",
    "Adj Close": "y_adj_close",
    "Volume": "y_volume",
    "Dividends": "dividends",
    "Stock Splits": "splits",
}

OUTPUT_COLS = [
    "ticker",
    "date",
    "open",
    "high",
    "low",
    "close",
    "close_raw",
    "volume",
    "volume_raw",
    "adj_factor",
    "dividends",
    "splits",
    "y_open",
    "y_high",
    "y_low",
    "y_close",
    "y_adj_close",
    "y_volume",
]


def future_split_factor(splits: pd.Series) -> pd.Series:
    """Product of split ratios strictly after each date (1.0 on and after the last split).

    A split effective on day s means prices from s onward are post-split, so the
    true price on any day t < s is the split-adjusted price times ratio(s).
    """
    ratio = splits.fillna(0.0).replace(0.0, 1.0).to_numpy(dtype=float)
    incl = np.cumprod(ratio[::-1])[::-1]  # product over dates >= t
    return pd.Series(np.append(incl[1:], 1.0), index=splits.index)


def build_price_frame(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Raw single-ticker Yahoo frame (DatetimeIndex) -> project price frame."""
    df = raw.rename(columns=YAHOO_COLS)
    for col in YAHOO_COLS.values():
        if col not in df:
            df[col] = 0.0 if col in ("dividends", "splits") else np.nan
    df = df[list(YAHOO_COLS.values())]
    df = df.dropna(subset=["y_open", "y_high", "y_low", "y_close"], how="all")

    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize().astype("datetime64[ns]")
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df[["dividends", "splits"]] = df[["dividends", "splits"]].fillna(0.0)

    adj_factor = df["y_adj_close"] / df["y_close"]
    fsf = future_split_factor(df["splits"])
    out = pd.DataFrame(
        {
            "ticker": ticker,
            "date": df.index,
            "open": df["y_open"] * adj_factor,
            "high": df["y_high"] * adj_factor,
            "low": df["y_low"] * adj_factor,
            "close": df["y_adj_close"],
            "close_raw": df["y_close"] * fsf,
            "volume": df["y_volume"].astype(float),
            "volume_raw": df["y_volume"].astype(float) / fsf,
            "adj_factor": adj_factor,
        },
        index=df.index,
    )
    out = pd.concat(
        [
            out,
            df[
                [
                    "dividends",
                    "splits",
                    "y_open",
                    "y_high",
                    "y_low",
                    "y_close",
                    "y_adj_close",
                    "y_volume",
                ]
            ].astype(float),
        ],
        axis=1,
    )
    return out[OUTPUT_COLS].reset_index(drop=True)
