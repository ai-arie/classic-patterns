"""Score today's pattern setups with the production bundle -> artifacts/screener/<date>.parquet.

python scripts/screen.py --refresh            # update prices, validate, clean, features, patterns
python scripts/screen.py                      # score the last session of the current build
python scripts/screen.py --date 2026-10-23    # an earlier session

One file per session. A later run of the same session never overwrites it: it is saved next to it
as <date>_<local run time>.parquet.

The app (``streamlit run src/classic_patterns/app/ui.py``) shows the latest file. It shows pattern
setups with estimated probabilities; you decide. Research / education, not financial advice.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time

import pandas as pd

from classic_patterns.app.scoring import Bundle
from classic_patterns.app.screener import result_path, screen
from classic_patterns.features.arrays import FeatureStore
from classic_patterns.utils.code_hash import project_root
from classic_patterns.utils.config import load_config, resolve
from classic_patterns.utils.logging import setup_logging

REFRESH = [
    ["scripts/update_daily.py"],
    ["scripts/validate_data.py"],
    ["scripts/build_clean.py"],
    ["scripts/build_features.py", "--no-report"],
    ["scripts/detect_patterns.py", "--steps", "build"],
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--refresh", action="store_true", help="run the daily data pipeline first")
    ap.add_argument("--date")
    ap.add_argument("--bundle", default="models/production")
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args()
    setup_logging(resolve("logs/screen.log"))
    if args.refresh:
        for step in REFRESH:
            t0 = time.perf_counter()
            subprocess.run([sys.executable, *step], cwd=project_root(), check=True)
            print(f"{step[0]}: {time.perf_counter() - t0:.0f} s")
    det = load_config("detector")
    store = FeatureStore(resolve(det["paths"]["features"]))
    events = pd.read_parquet(resolve(det["paths"]["out_dir"]) / "events.parquet")
    bundle = Bundle.load(resolve(args.bundle))
    res = screen(bundle, store, events, resolve(det["paths"]["prices"]), args.date, args.jobs)
    day = pd.Timestamp(args.date or store.rows["date"].max())
    out = resolve("artifacts/screener")
    out.mkdir(parents=True, exist_ok=True)
    path = result_path(out, day)
    res.to_parquet(path)
    logging.getLogger("screen").info("%s: %d setups -> %s", f"{day:%Y-%m-%d}", len(res), path.name)
    cols = ["ticker", "pattern", "stage", "close", "trigger", "stop", "lgbm_p_stop",
            "student_p_stop", "student_exp_r", "student_p_trigger20"]  # fmt: skip
    print(f"{day:%Y-%m-%d}: {len(res)} setups (research / education, not financial advice)")
    if len(res):
        print(res[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
