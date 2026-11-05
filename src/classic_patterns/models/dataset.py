"""Model context windows sliced on the fly from the training arrays (``features.arrays``).

A window for feature-store row ``g`` covers the ``L`` rows of the same ticker ending at ``g``
(inclusive), so it only uses data up to the close of that day. Rows before the ticker's first
row or before the row's context floor (``data.store.context_guards``) are NaN, and so are
bad-print rows (stock channels only). Chronos-2 treats NaN as missing.

Variates are named in the config: ``target`` = log(close / close at the window end), a stock
channel, ``log:<channel>`` for its log, or a market-context column (joined by date).
"""

from __future__ import annotations

import numpy as np

from classic_patterns.features.arrays import FeatureStore

TARGET = "target"


class WindowBuilder:
    def __init__(
        self,
        x: np.ndarray,
        channels: list[str],
        row_start: np.ndarray,
        floor: np.ndarray,
        bad: np.ndarray,
        ctx_idx: np.ndarray,
        market: np.ndarray,
        market_columns: list[str],
        variates: list[str],
    ):
        """``x``: [rows, channels] stock features; ``row_start``: global row of each row's ticker
        start; ``floor``: per-row context floor, local to the ticker; ``bad``: per-row bad-print
        flag; ``ctx_idx``: per-row market row; ``market``: [sessions, market_columns]."""
        self.variates = list(variates)
        stock, self._kinds = [], []
        for v in self.variates:
            name = v.removeprefix("log:")
            if v == TARGET:
                self._kinds.append(("stock", len(stock), False))
                stock.append("log_close")
            elif name in channels:
                self._kinds.append(("stock", len(stock), v.startswith("log:")))
                stock.append(name)
            elif name in market_columns:
                self._kinds.append(("market", market_columns.index(name), v.startswith("log:")))
            else:
                raise KeyError(f"unknown variate {v!r}")
        self.xs = np.ascontiguousarray(x[:, [channels.index(c) for c in stock]], dtype=np.float32)
        self.row_start = np.asarray(row_start, dtype=np.int64)
        self.first = self.row_start + np.asarray(floor, dtype=np.int64)  # first usable global row
        self.bad = np.asarray(bad, dtype=bool)
        self.ctx_idx = np.asarray(ctx_idx, dtype=np.int64)
        self.market = np.asarray(market, dtype=np.float32)

    @classmethod
    def from_store(cls, store: FeatureStore, variates: list[str]) -> WindowBuilder:
        rows, tick = store.rows, store.tickers
        offset = tick.set_index("ticker_id")["offset"].to_numpy()
        return cls(
            x=store.x,
            channels=store.channels,
            row_start=offset[rows["ticker_id"].to_numpy()],
            floor=rows["floor"].to_numpy(),
            bad=rows["bad"].to_numpy(),
            ctx_idx=rows["ctx_idx"].to_numpy(),
            market=store.market.drop(columns="date").to_numpy(dtype=np.float32),
            market_columns=list(store.market.columns.drop("date")),
            variates=variates,
        )

    def windows(self, g: np.ndarray, length: int) -> np.ndarray:
        """[len(g), n_variates, length] float32 windows ending at global rows ``g``."""
        g = np.asarray(g, dtype=np.int64)
        idx = g[:, None] + np.arange(-length + 1, 1)
        ok = idx >= self.first[g][:, None]
        safe = np.where(ok, idx, g[:, None])
        st = np.where((ok & ~self.bad[safe])[..., None], self.xs[safe], np.nan)
        mk = self.market[self.ctx_idx[safe]]
        mk[~ok] = np.nan
        out = np.empty((len(g), len(self.variates), length), dtype=np.float32)
        for j, (v, (kind, col, log)) in enumerate(zip(self.variates, self._kinds, strict=True)):
            w = st[..., col] if kind == "stock" else mk[..., col]
            if v == TARGET:
                w = w - w[:, -1:]
            elif log:
                with np.errstate(divide="ignore", invalid="ignore"):
                    w = np.log(np.where(w > 0, w, np.nan))
            out[:, j] = w
        return out
