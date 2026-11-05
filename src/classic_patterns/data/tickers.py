"""Ticker universe from the Nasdaq Trader symbol directory.

The universe is today's listings (survivorship bias is accepted and documented).
``in_sp500_today`` is for reporting only and must never filter historical data.
"""

from __future__ import annotations

import io
import logging
import re
from datetime import UTC, datetime

import pandas as pd
import requests

log = logging.getLogger(__name__)

USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) classic-patterns research"


def fetch_text(url: str, timeout: int = 60) -> str:
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def parse_symdir(text: str) -> pd.DataFrame:
    """Parse a pipe-delimited symbol directory file, dropping the trailer line."""
    lines = [ln for ln in text.splitlines() if ln and not ln.startswith("File Creation Time")]
    return pd.read_csv(io.StringIO("\n".join(lines)), sep="|", dtype=str, keep_default_na=False)


def to_yahoo_symbol(symbol: str) -> str:
    """Nasdaq/CQS symbol -> Yahoo symbol (class shares ``BRK.B`` -> ``BRK-B``)."""
    return symbol.replace(".", "-")


def normalize_listings(nasdaq: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    """Combine both files into: symbol, name, exchange, is_etf, is_test, source_file."""
    a = pd.DataFrame(
        {
            "symbol": nasdaq["Symbol"].str.strip(),
            "name": nasdaq["Security Name"].str.strip(),
            "exchange": "Q",
            "is_etf": nasdaq["ETF"].eq("Y"),
            "is_test": nasdaq["Test Issue"].eq("Y"),
            "source_file": "nasdaqlisted",
        }
    )
    b = pd.DataFrame(
        {
            "symbol": other["CQS Symbol"].str.strip(),
            "name": other["Security Name"].str.strip(),
            "exchange": other["Exchange"].str.strip(),
            "is_etf": other["ETF"].eq("Y"),
            "is_test": other["Test Issue"].eq("Y"),
            "source_file": "otherlisted",
        }
    )
    return pd.concat([a, b], ignore_index=True)


def drop_reason(
    row: pd.Series, cfg: dict, name_patterns: dict[str, re.Pattern], other_venue_keep: re.Pattern
) -> str:
    if row["is_test"]:
        return "test_issue"
    if row["is_etf"]:
        return "etf"
    if row["exchange"] != "Q" and row["exchange"] not in cfg["keep_exchanges"]:
        if not other_venue_keep.search(row["name"]):
            return f"exchange_{row['exchange']}"
    if "$" in row["symbol"]:
        return "preferred_symbol"
    lower = re.findall(r"[a-z]", row["symbol"])
    if lower:
        return cfg["cqs_lowercase_reasons"].get(lower[0], "special_symbol")
    if re.search(cfg["never_drop_name_pattern"], row["name"], re.IGNORECASE):
        return ""
    for reason, pat in name_patterns.items():
        if pat.search(row["name"]):
            return reason
    return ""


def filter_universe(listings: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (kept, dropped); ``dropped`` has a ``reason`` column."""
    patterns = {k: re.compile(v, re.IGNORECASE) for k, v in cfg["drop_name_patterns"].items()}
    other_venue_keep = re.compile(cfg["other_venue_keep_pattern"], re.IGNORECASE)
    reasons = listings.apply(
        drop_reason, axis=1, cfg=cfg, name_patterns=patterns, other_venue_keep=other_venue_keep
    )
    dropped = listings[reasons != ""].assign(reason=reasons[reasons != ""])
    kept = listings[reasons == ""].copy()

    adr = re.compile(cfg["adr_pattern"], re.IGNORECASE)
    spac = re.compile(cfg["spac_pattern"], re.IGNORECASE)
    kept["is_adr"] = kept["name"].map(lambda s: bool(adr.search(s)))
    kept["is_spac"] = kept["name"].map(lambda s: bool(spac.search(s)))
    kept["yahoo_symbol"] = kept["symbol"].map(to_yahoo_symbol)

    dup = kept["yahoo_symbol"].duplicated(keep="first")
    if dup.any():
        log.warning("dropping %d duplicate yahoo symbols", int(dup.sum()))
        dropped = pd.concat([dropped, kept[dup].assign(reason="duplicate_symbol")])
        kept = kept[~dup]
    return kept.reset_index(drop=True), dropped.reset_index(drop=True)


def fetch_sp500_symbols(url: str) -> set[str]:
    tables = pd.read_html(io.StringIO(fetch_text(url)), attrs={"id": "constituents"})
    return {to_yahoo_symbol(s.strip()) for s in tables[0]["Symbol"]}


def build_universe(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    src = cfg["sources"]
    retrieved_at = datetime.now(UTC).replace(tzinfo=None)
    nasdaq = parse_symdir(fetch_text(src["nasdaqlisted"]))
    other = parse_symdir(fetch_text(src["otherlisted"]))
    listings = normalize_listings(nasdaq, other)
    kept, dropped = filter_universe(listings, cfg)

    sp500 = fetch_sp500_symbols(src["sp500_wikipedia"])
    kept["in_sp500_today"] = kept["yahoo_symbol"].isin(sp500)
    missing = sorted(sp500 - set(kept["yahoo_symbol"]))
    if missing:
        log.warning("S&P 500 members not in universe: %s", missing)
    kept["retrieved_at"] = retrieved_at
    dropped["retrieved_at"] = retrieved_at

    cols = [
        "symbol",
        "yahoo_symbol",
        "name",
        "exchange",
        "is_adr",
        "is_spac",
        "in_sp500_today",
        "source_file",
        "retrieved_at",
    ]
    return kept[cols], dropped
