"""Read access to the training arrays written by ``features.build`` (no windows materialized)."""

from __future__ import annotations

import json
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd


class FeatureStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.meta = json.loads((self.path / "meta.json").read_text())
        self.channels: list[str] = self.meta["channels"]
        self.x = np.load(self.path / "stock_features.npy", mmap_mode="r")

    @cached_property
    def rows(self) -> pd.DataFrame:
        return pd.read_parquet(self.path / "rows.parquet")

    @cached_property
    def tickers(self) -> pd.DataFrame:
        return pd.read_parquet(self.path / "tickers.parquet")

    @cached_property
    def market(self) -> pd.DataFrame:
        return pd.read_parquet(self.path / "market.parquet")

    def channel(self, name: str) -> int:
        return self.channels.index(name)

    def ticker_slice(self, ticker: str) -> slice:
        t = self.tickers.set_index("ticker").loc[ticker]
        return slice(int(t["offset"]), int(t["offset"] + t["n_rows"]))

    def frame(self, ticker: str, with_market: bool = False) -> pd.DataFrame:
        """One ticker's rows, features (and optionally market context) as a DataFrame."""
        sl = self.ticker_slice(ticker)
        df = self.rows.iloc[sl].reset_index(drop=True)
        df = pd.concat([df, pd.DataFrame(np.asarray(self.x[sl]), columns=self.channels)], axis=1)
        if with_market:
            m = self.market.drop(columns="date").iloc[df["ctx_idx"].to_numpy()]
            df = pd.concat([df, m.reset_index(drop=True)], axis=1)
        return df

    def eligible(self, bit: int = 0) -> np.ndarray:
        """Per-row eligibility for dollar-volume threshold ``meta['elig_bits'][bit]``."""
        return (self.rows["elig"].to_numpy() >> bit & 1).astype(bool)


class EdgarBlock:
    """Read access to the optional extra feature block: float32
    [total_rows, channels] aligned with the stock feature store rows, channel groups in meta."""

    def __init__(self, path: str | Path, store: FeatureStore | None = None):
        self.path = Path(path)
        self.meta = json.loads((self.path / "meta.json").read_text())
        self.channels: list[str] = self.meta["channels"]
        self.groups: dict[str, list[str]] = self.meta["groups"]
        self.x = np.load(self.path / "x.npy", mmap_mode="r")
        if store is not None and self.meta["feature_store_created_at"] != store.meta["created_at"]:
            raise ValueError("EDGAR block was built for a different feature store; rebuild it")

    def frame(self, fs_rows: np.ndarray, groups) -> pd.DataFrame:
        """Columns of ``groups`` (e.g. ("fund", "earn", "sector")) at feature-store rows."""
        cols = [c for g in groups for c in self.groups[g]]
        idx = [self.channels.index(c) for c in cols]
        g = np.asarray(fs_rows, dtype=np.int64)
        order = np.argsort(g)
        x = np.empty((len(g), len(idx)), dtype=np.float32)
        x[order] = np.asarray(self.x[g[order]])[:, idx]
        return pd.DataFrame(x, columns=cols)
