"""Build features, eligibility and training arrays (artifacts/features) and the B1 report.

python scripts/build_features.py                    # full build + report
python scripts/build_features.py --tickers AAPL MSFT --out-dir artifacts/features_test
python scripts/build_features.py --report-only      # rewrite the report from an existing build
"""

from __future__ import annotations

import argparse
import logging
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from classic_patterns.features.arrays import FeatureStore  # noqa: E402
from classic_patterns.features.build import build  # noqa: E402
from classic_patterns.utils.code_hash import project_root  # noqa: E402
from classic_patterns.utils.config import load_config, resolve  # noqa: E402
from classic_patterns.utils.logging import setup_logging  # noqa: E402

log = logging.getLogger("build_features")


def _fmt_int(x) -> str:
    return f"{int(round(x)):,}"


def md_table(df: pd.DataFrame) -> list[str]:
    cols = [str(df.index.name or ""), *map(str, df.columns)]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += [
        "| " + " | ".join([str(i), *map(str, r)]) + " |"
        for i, r in zip(df.index, df.values, strict=True)
    ]
    return out


def filter_table(fs: pd.DataFrame, thr: float) -> pd.DataFrame:
    f = fs[fs["min_dollar_volume"] == thr].set_index("year")
    t = pd.DataFrame(index=f.index)
    t["stock-days"] = f["rows"]
    for k, name in (
        ("fail_price", "< $5"),
        ("fail_dollar_volume", "low $ volume"),
        ("fail_history", "< 300 days history"),
        ("in_span", "exclusion span"),
    ):
        t[name] = f[k]
    t["eligible (no hysteresis)"] = f["eligible_raw"]
    t["eligible"] = f["eligible"]
    t["eligible %"] = 100 * f["eligible"] / f["rows"]
    return t


def write_report(store: FeatureStore, cfg: dict, runtime_s: float | None) -> None:
    root = project_root()
    meta = store.meta
    fig_dir = resolve(cfg["paths"]["figures"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    fs = pd.read_parquet(store.path / "filter_stats.parquet")
    bits = meta["elig_bits"]
    start = cfg["build"]["report_start"]
    rows = store.rows
    sel = (rows["date"] >= start).to_numpy()

    lines = ["# B1 — Features, eligibility and training arrays", ""]
    lines += [
        f"Build: {meta['created_at']}, code `{meta['code_hash'][:19]}`, data manifest "
        f"`{meta['data_manifest_sha256'][:12]}`.",
        "",
    ]
    size_gb = (store.path / "stock_features.npy").stat().st_size / 1e9
    lines += [
        f"- {meta['n_tickers']:,} tickers, {meta['n_rows']:,} rows "
        f"({meta['first_date']} → {meta['last_date']}), {len(store.channels)} stock "
        f"channels (float32 memmap, {size_gb:.2f} GB), {len(meta['market_columns'])} "
        f"market-context columns over {len(store.market):,} sessions."
    ]
    if runtime_s is None:
        runtime_s = meta.get("build_seconds")
    if runtime_s is not None:
        lines += [f"- Build time: {runtime_s / 60:.1f} min with {cfg['build']['workers']} workers."]
    lines += [
        "- Truncation invariance (features computed on data cut at *t* equal the full-history "
        "values at *t*) is tested for every stock, weekly and market feature and for "
        "eligibility, on synthetic series and on random real tickers/dates "
        "(`tests/test_features_truncation.py`).",
        "",
    ]

    # eligibility per year
    lines += [
        "## Eligibility filters per year",
        "",
        f"Default rule: close_raw ≥ ${cfg['eligibility']['min_price']:.0f}, 50-day dollar "
        f"volume ≥ ${bits[0] / 1e6:.0f}M, ≥ {cfg['eligibility']['min_history']} days of "
        f"history, not in an exclusion span; hysteresis {cfg['eligibility']['off_days']} "
        "days. Filters overlap, so the columns don't add up. Stock-days = all rows in "
        "the data (stocks only exist in the data while listed; survivorship, see below).",
        "",
    ]
    t = filter_table(fs, bits[0])
    tf = t.copy()
    for c in tf.columns:
        tf[c] = tf[c].map(lambda v, c=c: f"{v:.1f}" if c.endswith("%") else _fmt_int(v))
    lines += [*md_table(tf), ""]

    tot = fs.groupby("min_dollar_volume")[["rows", "eligible"]].sum()
    lines += [
        f"Eligible stock-days by dollar-volume threshold (all years from {start[:4]}):",
        "",
        "| threshold | eligible stock-days | share |",
        "|---|---|---|",
    ]
    for thr, r in tot.sort_index().iterrows():
        lines += [
            f"| ${thr / 1e6:.0f}M | {_fmt_int(r['eligible'])} | "
            f"{100 * r['eligible'] / r['rows']:.1f}% |"
        ]
    lines += [""]
    hyst_gain = t["eligible"].sum() / t["eligible (no hysteresis)"].sum() - 1
    lines += [f"Hysteresis adds {100 * hyst_gain:.1f}% eligible stock-days vs the plain rule.", ""]

    # eligible stocks per day + breadth figure
    m = store.market.set_index("date")
    m = m[m.index >= start]
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for k, thr in enumerate(bits):
        e = store.eligible(k)[sel]
        n = pd.Series(e, index=rows.loc[sel, "date"]).groupby(level=0).sum()
        ax[0].plot(n.index, n.to_numpy(), lw=0.8, label=f"${thr / 1e6:.0f}M")
    ax[0].set_ylabel("eligible stocks / day")
    ax[0].legend()
    ax[1].plot(m.index, 100 * m["breadth_sma200"], lw=0.6, label="% above SMA200")
    ax[1].plot(m.index, 100 * m["breadth_sma50"], lw=0.4, alpha=0.7, label="% above SMA50")
    ax[1].set_ylabel("breadth (%)")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "eligible_and_breadth.png", dpi=120)
    plt.close(fig)
    rel = (fig_dir / "eligible_and_breadth.png").relative_to(root / "reports")
    ne = m["n_eligible"]
    lines += [
        "## Eligible stocks per day and breadth",
        "",
        f"![eligible and breadth]({rel})",
        "",
        f"Eligible stocks per day (default rule): {ne.iloc[0]:.0f} on {ne.index[0].date()}, "
        f"median {ne.median():.0f}, max {ne.max():.0f}, {ne.iloc[-1]:.0f} on "
        f"{ne.index[-1].date()}.",
        "",
    ]

    # feature summary on eligible rows
    e0 = store.eligible(0) & sel
    idx = np.flatnonzero(e0)
    rng = np.random.default_rng(0)
    samp = np.sort(rng.choice(idx, size=min(500_000, len(idx)), replace=False))
    x = pd.DataFrame(np.asarray(store.x[samp]), columns=store.channels)
    q = x.quantile([0.01, 0.5, 0.99]).T
    q.insert(0, "NaN %", 100 * x.isna().mean())
    lines += [
        "## Stock channels (eligible rows, 500k random sample)",
        "",
        "| channel | NaN % | p1 | median | p99 |",
        "|---|---|---|---|---|",
    ]
    for name, r in q.iterrows():
        lines += [f"| `{name}` | {r['NaN %']:.2f} | {r[0.01]:.4g} | {r[0.5]:.4g} | {r[0.99]:.4g} |"]
    lines += [
        "",
        f"`big_gap` is set on {100 * np.nanmean(x['big_gap']):.2f}% of eligible days.",
        "",
    ]
    mm = m.drop(columns=["n_eligible"])
    lines += [
        "## Market context (sessions from " + start[:4] + ")",
        "",
        "| column | first valid | NaN % | median |",
        "|---|---|---|---|",
    ]
    for c in mm.columns:
        s = mm[c]
        fv = s.first_valid_index()
        lines += [
            f"| `{c}` | {fv.date() if fv is not None else '-'} | "
            f"{100 * s.isna().mean():.1f} | {s.median():.4g} |"
        ]
    lines += [""]

    lines += [
        "## Definitions",
        "",
        "- All price features use split/dividend-adjusted OHLC and are scale-free (ratios, "
        "%, ATR units), so a later re-adjustment of the history doesn't change them.",
        "- `vol_ratio50` = volume / mean volume of the previous 50 days; "
        "`log_dollar_vol50` = log(1 + 50-day mean of close_raw × volume_raw).",
        "- `rs_level` = log(RS / SMA200(RS)) with RS = close / ^GSPC; `rs_chg63` = 3-month "
        'log change of RS (the plan\'s "ratio" made scale-free).',
        "- `big_gap`: |open / previous close − 1| > 3 × previous day's ATR% (earnings proxy).",
        "- Weekly features use Friday-ending weeks and only weeks whose last NYSE session is ≤ t.",
        "- Market context: each index computed on its own rows, carried forward ≤ 5 "
        "sessions. `vix_term` = VIX / VIX3M (same day only; VIX3M starts 2006-07). "
        "`t10y2y` usable from the session after its FRED date. HY OAS not used (FRED "
        "history only from 2023-10).",
        "- Breadth = share of default-eligible stocks with close above their SMA200 / "
        "SMA50, computed point-in-time from this dataset.",
        "- Context guards (`floor`, `bad`) from the A3 exclusions are stored per row.",
        "",
    ]
    lines += [
        "## Limitations",
        "",
        "- **Survivorship bias:** the universe comes from today's listings; stocks that were "
        "delisted before the download are missing, so early years hold far fewer stocks "
        "and a winner-tilted sample.",
        "- **yfinance limitations:** adjusted prices and volumes come from Yahoo and carry "
        "its errors (A3 masks the detected ones).",
        "",
    ]
    out = resolve(cfg["paths"]["report"])
    out.write_text("\n".join(lines))
    log.info("report: %s", out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="features")
    ap.add_argument("--tickers", nargs="*")
    ap.add_argument("--out-dir")
    ap.add_argument("--workers", type=int)
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--no-report", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.out_dir:
        cfg["paths"]["out_dir"] = args.out_dir
    setup_logging(resolve(cfg["paths"]["log_file"]))
    runtime = None
    if not args.report_only:
        t0 = time.perf_counter()
        build(cfg, project_root(), workers=args.workers, tickers=args.tickers)
        runtime = time.perf_counter() - t0
        log.info("build done in %.1f min", runtime / 60)
    if not args.no_report and not args.tickers:
        write_report(FeatureStore(resolve(cfg["paths"]["out_dir"])), cfg, runtime)


if __name__ == "__main__":
    main()
