"""Setup charts (candles, trigger / stop / target, pattern points) for the app and the digest."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from classic_patterns.utils.config import load_config, resolve

PIVOTS = ("start_row", "p1_row", "p2_row", "p3_row", "p4_row", "p5_row", "head_row")


def load_prices(tickers: str | list[str]) -> pd.DataFrame:
    """Daily OHLC of one ticker (index = the ticker's row number, as the pivot columns use) or
    of several (with a ``ticker`` column; split with :func:`ticker_prices`)."""
    p = resolve(load_config("detector")["paths"]["prices"])
    cols = ["ticker", "date", "open", "high", "low", "close"]
    if isinstance(tickers, str):
        px = pd.read_parquet(p, columns=cols, filters=[("ticker", "==", tickers)])
        return px.drop(columns="ticker").reset_index(drop=True)
    return pd.read_parquet(p, columns=cols, filters=[("ticker", "in", list(tickers))])


def ticker_prices(px: pd.DataFrame, ticker: str) -> pd.DataFrame:
    return px[px["ticker"] == ticker].drop(columns="ticker").reset_index(drop=True)


def chart(row: pd.Series, px: pd.DataFrame, days: int, height: int = 480) -> go.Figure:
    """``px``: the ticker's prices from :func:`load_prices` / :func:`ticker_prices`."""
    end = int(px.index[px["date"] == row["date"]][0])
    start = max(0, min(end - days, int(row.get("start_row", end - days)) - 20))
    w = px.iloc[start : end + 1]
    fig = go.Figure(go.Candlestick(x=w["date"], open=w["open"], high=w["high"], low=w["low"],
                                   close=w["close"], name=row["ticker"]))  # fmt: skip
    for name, y, color in (("trigger", row["trigger"], "#888"), ("stop", row["stop"], "#c0392b"),
                           ("target", row["target"], "#27ae60")):  # fmt: skip
        fig.add_hline(y=y, line_dash="dash", line_color=color, annotation_text=name)
    piv = [int(row[c]) for c in PIVOTS if c in row and pd.notna(row[c]) and row[c] >= start]
    if piv:
        pts = px.iloc[sorted(set(piv))]
        fig.add_scatter(x=pts["date"], y=pts["close"], mode="markers", name="pattern points",
                        marker={"size": 9, "color": "#2c3e50"})  # fmt: skip
    fig.update_layout(
        height=height,
        xaxis_rangeslider_visible=False,
        margin={"t": 30, "b": 10},
        title=f"{row['ticker']} — {row['pattern']} ({row['stage']}, {row['side']})",
    )
    return fig
