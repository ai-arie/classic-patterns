"""Check that this machine's data build and app scoring reproduce the bundle's parity sample.

The bundle stores ~300 historical setups scored on the training machine, without prices. Here the
same setups are looked up in the local events table, re-scored with the app code and compared.
Setups are matched by their scale-free geometry, because Yahoo re-adjusts prices after dividends
and splits (see ``classic_patterns.app.parity``).

Result:
  PARITY OK                 every setup found, LightGBM within 1e-6, int8 student within 0.02
  PARITY OK (data drift)    a few setups missing or slightly off because the data changed
                            (exit 0; details in artifacts/validation/parity_details.csv)
  PARITY FAILED             too many setups missing or off, or a large difference (exit 1)

python scripts/parity_check.py [--bundle models/production]
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from classic_patterns.app.parity import (
    COMPARED,
    DRIFT_TOL,
    KEY,
    MODELS,
    TOL,
    match,
    verdict,
)
from classic_patterns.app.scoring import Bundle, close_atr, score_events
from classic_patterns.features.arrays import FeatureStore
from classic_patterns.utils.config import load_config, resolve


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bundle", default="models/production")
    ap.add_argument("--out", default="artifacts/validation/parity_details.csv")
    args = ap.parse_args()
    bundle = Bundle.load(resolve(args.bundle))
    ref = pd.read_parquet(bundle.root / "parity.parquet").reset_index(drop=True)
    det = load_config("detector")
    store = FeatureStore(resolve(det["paths"]["features"]))
    events = pd.read_parquet(resolve(det["paths"]["out_dir"]) / "events.parquet")

    ev = events.reset_index(drop=True)
    pairs = ref[KEY].reset_index(names="ref_i").merge(ev[KEY].reset_index(names="ev_i"), on=KEY)
    e = ev.iloc[pairs["ev_i"].to_numpy()].reset_index(drop=True)
    close, atr = close_atr(store, resolve(det["paths"]["prices"]), e["fs_row"].to_numpy())
    s = score_events(bundle, e, store, close, atr)
    cand = pd.concat([pairs[["ref_i"]], s.reset_index(drop=True)], axis=1)
    m = match(ref, cand)

    counts = m["status"].value_counts()
    print(f"parity sample: {len(m)} setups")
    for st, note in (
        ("matched", "same geometry as on the training machine: compared"),
        ("changed", "still detected, different geometry (revised data): not compared"),
        ("missing", "not detected locally any more: not compared"),
    ):
        print(f"  {st:<11}{counts.get(st, 0):>4}  {note}")
    comp = m[m["status"].isin(COMPARED)]
    for k in MODELS:
        d = comp[f"d_{k}"]
        n_in = int((d <= TOL[k]).sum())
        print(
            f"{k}: {n_in}/{len(comp)} within {TOL[k]:g}, max |dp| {d.max():.2e} "
            f"(data-drift limit {DRIFT_TOL[k]:g})"
        )

    off = ~m["status"].isin(COMPARED)
    for k in MODELS:
        off |= m[f"d_{k}"] > TOL[k]
    if off.any():
        cols = [*KEY, "status", "d_lgbm", "d_student"]
        with pd.option_context("display.width", 200, "display.max_rows", 100):
            print("\nnot reproduced exactly:")
            print(m.loc[off, cols].to_string(index=False, float_format=lambda x: f"{x:.3g}"))
    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    m.to_csv(out, index=False)
    print(f"\ndetails: {out}")

    status, why = verdict(m)
    for w in why:
        print(f"  - {w}")
    print({"OK": "PARITY OK", "DRIFT": "PARITY OK (data drift)", "FAILED": "PARITY FAILED"}[status])
    sys.exit(1 if status == "FAILED" else 0)


if __name__ == "__main__":
    main()
