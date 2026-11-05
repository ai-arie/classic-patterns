"""Build the ticker universe: data/meta/universe.parquet (+ universe_dropped.parquet)."""

from __future__ import annotations

import argparse
import logging

from classic_patterns.data.tickers import build_universe
from classic_patterns.utils.config import load_config, resolve
from classic_patterns.utils.logging import setup_logging

log = logging.getLogger("build_universe")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="universe")
    args = ap.parse_args()
    setup_logging(resolve("logs/universe.log"))

    kept, dropped = build_universe(load_config(args.config))
    out = resolve("data/meta")
    out.mkdir(parents=True, exist_ok=True)
    kept.to_parquet(out / "universe.parquet", compression="zstd", index=False)
    dropped.to_parquet(out / "universe_dropped.parquet", compression="zstd", index=False)

    log.info(
        "kept %d tickers by exchange: %s", len(kept), kept["exchange"].value_counts().to_dict()
    )
    log.info(
        "ADR=%d SPAC=%d in_sp500_today=%d",
        kept["is_adr"].sum(),
        kept["is_spac"].sum(),
        kept["in_sp500_today"].sum(),
    )
    log.info("dropped %d by reason: %s", len(dropped), dropped["reason"].value_counts().to_dict())


if __name__ == "__main__":
    main()
