"""Market context: one value per NYSE session, the same for every stock on that date."""

from __future__ import annotations

import numpy as np
import pandas as pd

from classic_patterns.features import indicators as ind


def _series(bench: pd.DataFrame, ticker: str) -> pd.DataFrame:
    return bench[bench["ticker"] == ticker].sort_values("date").reset_index(drop=True)


def index_features(df: pd.DataFrame, trend_cfg: dict, rv_window: int) -> pd.DataFrame:
    """Trend block + realized volatility for one index, indexed by its own dates."""
    c = df["close"].to_numpy(dtype=np.float64)
    out = ind.trend_block(c, trend_cfg)
    r = np.log(ind.ratio(c, ind.shift(c, 1)))
    out[f"rv{rv_window}"] = ind.rolling_std(r, rv_window) * np.sqrt(252.0)
    return pd.DataFrame(out, index=pd.DatetimeIndex(df["date"]))


def vix_features(vix: pd.DataFrame, vix3m: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    v = vix["close"].to_numpy(dtype=np.float64)
    out = pd.DataFrame(
        {
            "vix": v,
            f"vix_pct{cfg['vix_percentile_window']}": ind.rolling_percentile(
                v, cfg["vix_percentile_window"]
            ),
            f"vix_chg{cfg['vix_change']}": np.log(ind.ratio(v, ind.shift(v, cfg["vix_change"]))),
        },
        index=pd.DatetimeIndex(vix["date"]),
    )
    v3 = vix3m.set_index("date")["close"].reindex(out.index)  # same-day values only
    out["vix_term"] = ind.ratio(v, v3.to_numpy(dtype=np.float64))
    return out


def macro_features(macro: pd.DataFrame, sessions: pd.DatetimeIndex, cfg: dict) -> pd.DataFrame:
    """FRED series as of each session, respecting the publication lag: a value dated d is
    usable from the ``macro_lag_sessions``-th session after d."""
    out = {}
    lag = cfg["macro_lag_sessions"]
    for name in cfg["macro"]:
        s = macro[macro["series"] == name].dropna(subset=["value"]).sort_values("date")
        # first session strictly after the value's date, then lag - 1 more sessions
        pos = np.searchsorted(sessions.values, s["date"].values, side="right") + (lag - 1)
        avail = pd.Series(s["value"].to_numpy(), index=pos)
        avail = avail[avail.index < len(sessions)]
        avail = avail[~avail.index.duplicated(keep="last")]
        col = np.full(len(sessions), np.nan)
        col[avail.index.to_numpy()] = avail.to_numpy()
        out[name.lower()] = pd.Series(col).ffill().to_numpy()
    return pd.DataFrame(out, index=sessions)


def market_context(
    bench: pd.DataFrame, macro: pd.DataFrame, sessions: pd.DatetimeIndex, cfg: dict, trend_cfg: dict
) -> pd.DataFrame:
    """All market-context columns except breadth, indexed by ``sessions``.

    Each index is computed on its own rows, then aligned to the sessions by carrying the last
    value forward for at most ``ffill_limit`` sessions.
    """
    parts = []
    for prefix, ticker in cfg["indices"].items():
        f = index_features(_series(bench, ticker), trend_cfg, cfg["realized_vol_window"])
        parts.append(f.add_prefix(f"{prefix}_"))
    parts.append(vix_features(_series(bench, cfg["vix"]), _series(bench, cfg["vix3m"]), cfg))
    aligned = [
        p.reindex(p.index.union(sessions)).ffill(limit=cfg["ffill_limit"]).reindex(sessions)
        for p in parts
    ]
    out = pd.concat(aligned, axis=1)
    if cfg.get("macro"):
        out = out.join(macro_features(macro, sessions, cfg))
    out.index.name = "date"
    return out


def breadth(counts: pd.DataFrame) -> pd.DataFrame:
    """% of eligible stocks above their SMA200 / SMA50, from per-session counts with columns
    n_eligible, n_valid{n}, n_above{n}."""
    out = pd.DataFrame(index=counts.index)
    out["n_eligible"] = counts["n_eligible"]
    for n in (200, 50):
        out[f"breadth_sma{n}"] = ind.ratio(counts[f"n_above{n}"], counts[f"n_valid{n}"])
    return out
