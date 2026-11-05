"""Pattern detection for the whole dataset, the hand-labeling sample, the gallery and the B2 report.

python scripts/detect_patterns.py                   # build + sample + gallery + report
python scripts/detect_patterns.py --steps report    # any subset of: build sample gallery report
python scripts/detect_patterns.py --tickers AAPL MSFT --out-dir artifacts/patterns_test \
    --steps build
"""

from __future__ import annotations

import argparse
import html
import json
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from classic_patterns.patterns.build import build  # noqa: E402
from classic_patterns.patterns.common import PATTERNS, STAGES  # noqa: E402
from classic_patterns.patterns.labeling import (  # noqa: E402
    draw_sample,
    load_labels,
    refresh_sample,
    top_up_sample,
)
from classic_patterns.patterns.render import (  # noqa: E402
    PATTERN_TEXT,
    measured_values,
    pattern_figure,
)
from classic_patterns.utils.code_hash import project_root  # noqa: E402
from classic_patterns.utils.config import load_config, resolve  # noqa: E402
from classic_patterns.utils.logging import setup_logging  # noqa: E402

log = logging.getLogger("detect_patterns")
HOLDOUT = pd.Timestamp("2024-01-01")
LIQ_BINS = [0, 15e6, 30e6, 100e6, 1e9, np.inf]
LIQ_NAMES = ["< $15M", "$15–30M", "$30–100M", "$100M–1B", "≥ $1B"]


def md_table(df: pd.DataFrame) -> list[str]:
    cols = [str(df.index.name or ""), *map(str, df.columns)]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += [
        "| " + " | ".join([str(i), *map(str, r)]) + " |"
        for i, r in zip(df.index, df.values, strict=True)
    ]
    return out


def _fmt(df: pd.DataFrame) -> pd.DataFrame:
    return df.map(lambda v: f"{int(v):,}" if isinstance(v, int | np.integer) else v)


def load_bars(cfg: dict, tickers) -> dict[str, pd.DataFrame]:
    p = pd.read_parquet(
        resolve(cfg["paths"]["prices"]),
        columns=["ticker", "date", "open", "high", "low", "close", "volume"],
        filters=[("ticker", "in", sorted(set(tickers)))],
    )
    return {t: g.sort_values("date").reset_index(drop=True) for t, g in p.groupby("ticker")}


# ---------------------------------------------------------------- sample


def make_sample(cfg: dict) -> pd.DataFrame:
    out_dir = resolve(cfg["paths"]["out_dir"])
    loose = pd.read_parquet(out_dir / "events_loose.parquet")
    path = out_dir / "label_sample.parquet"
    if path.exists() and len(load_labels(resolve(cfg["paths"]["labels"]))):
        s = refresh_sample(pd.read_parquet(path), loose)
        log.info("labeling sample: %d events kept (labels exist), columns refreshed", len(s))
    else:
        s = draw_sample(loose, cfg)
        log.info("labeling sample: %d events drawn", len(s))
    n = len(s)
    s = top_up_sample(s, loose, cfg, load_labels(resolve(cfg["paths"]["labels"])))
    if len(s) > n:
        log.info("labeling sample: %d top-up events added (sample_id %d+)", len(s) - n, n + 1)
    s.to_parquet(path)
    return s


# ---------------------------------------------------------------- gallery


def make_gallery(cfg: dict) -> dict[str, pd.DataFrame]:
    gc = cfg["gallery"]
    out_dir = resolve(cfg["paths"]["out_dir"])
    gal_dir = resolve(cfg["paths"]["gallery"])
    gal_dir.mkdir(parents=True, exist_ok=True)
    ev = pd.read_parquet(out_dir / "events.parquet")
    ev = ev[
        (ev["stage"] == "triggered")
        & ((ev["elig"] & 1) == 1)
        & ~ev["data_flag"]
        & (ev["date"] >= cfg["build"]["report_start"])
        & (ev["date"] <= gc["end"])
    ]
    rng = np.random.default_rng(gc["seed"])
    picks = {}
    for pat in PATTERNS:
        cell = ev[ev["pattern"] == pat]
        idx = rng.choice(len(cell), size=min(gc["n_per_pattern"], len(cell)), replace=False)
        picks[pat] = cell.iloc[np.sort(idx)].sort_values("date").reset_index(drop=True)
    bars = load_bars(cfg, pd.concat(picks.values())["ticker"])
    for pat, sel in picks.items():
        blocks = []
        for i, e in sel.iterrows():
            b = bars[e["ticker"]]
            assert b["date"].iloc[int(e["row"])] == e["date"]
            fig = pattern_figure(
                b,
                e,
                before=cfg["sample"]["context_days"],
                after=gc["after_days"],
                show_levels=True,
                height=520,
            )
            fig.update_layout(title=f"{i + 1}. {e['ticker']} — triggered {e['date']:%Y-%m-%d}")
            vals = measured_values(e) + [
                ("Stop (textbook)", f"{e['stop']:.2f}"),
                ("Target (textbook)", f"{e['target']:.2f}"),
            ]
            rows = "".join(
                f"<tr><td>{html.escape(k)}</td><td>{html.escape(v)}</td></tr>" for k, v in vals
            )
            blocks.append(
                '<div class="ev"><div class="chart">'
                + fig.to_html(full_html=False, include_plotlyjs=False)
                + f'</div><table class="vals">{rows}</table></div>'
            )
        page = _page(
            f"{PATTERN_TEXT[pat]} — {len(sel)} random detections",
            "<p>Random triggered detections (default thresholds, eligible stock-days, trigger "
            f"dates {cfg['build']['report_start']} → {gc['end']}). Dark candles: up to and "
            f"including the trigger day. Blue candles: the next {gc['after_days']} trading days. "
            "Dashed lines: textbook stop and target. Prices are split/dividend-adjusted.</p>"
            + "".join(blocks),
        )
        (gal_dir / f"{pat}.html").write_text(page)
        log.info("gallery %s: %d charts", pat, len(sel))
    links = "".join(
        f'<li><a href="{p}.html">{PATTERN_TEXT[p]}</a> ({len(picks[p])})</li>' for p in PATTERNS
    )
    (gal_dir / "index.html").write_text(_page("Pattern gallery", f"<ul>{links}</ul>"))
    pd.concat(picks.values()).to_csv(gal_dir / "gallery_events.csv", index=False)
    return picks


def _page(title: str, body: str) -> str:
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
body {{ font-family: system-ui, sans-serif; margin: 16px; color: #222; background: #fff; }}
.ev {{ display: flex; flex-wrap: wrap; gap: 12px; border-top: 1px solid #ddd; padding: 8px 0; }}
.chart {{ flex: 1 1 640px; min-width: 0; }}
table.vals {{ font-size: 13px; border-collapse: collapse; align-self: flex-start; }}
table.vals td {{ padding: 2px 8px; border-bottom: 1px solid #eee; }}
</style></head><body><h1>{html.escape(title)}</h1>
<p><em>Research project, not financial advice.</em></p>{body}</body></html>"""


# ---------------------------------------------------------------- report


def write_report(cfg: dict) -> None:
    out_dir = resolve(cfg["paths"]["out_dir"])
    fig_dir = resolve(cfg["paths"]["figures"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    meta = json.loads((out_dir / "meta.json").read_text())
    ev = pd.read_parquet(out_dir / "events.parquet")
    loose = pd.read_parquet(out_dir / "events_loose.parquet")
    start = pd.Timestamp(cfg["build"]["report_start"])
    elig = ((ev["elig"] & 1) == 1) & ~ev["data_flag"] & (ev["date"] >= start)
    L = ["# B2 — Pattern detectors", ""]
    L += [
        f"Build: {meta['created_at']}, code `{meta['code_hash'][:19]}`, data manifest "
        f"`{meta['data_manifest_sha256'][:12]}`, {meta['n_tickers']:,} tickers, "
        f"{meta['build_seconds'] / 60:.1f} min on {cfg['build']['workers']} workers "
        "(default + loose thresholds).",
        "",
        "Detectors: `src/classic_patterns/patterns/` (pivots, cup_handle, head_shoulders, detect); "
        "thresholds in `configs/detector.yaml`. One row per pattern instance and stage, dated on "
        "the day the stage is reached (all inputs ≤ that day). *Eligible* = default liquidity "
        "rule ($5, $15M 50-day dollar volume, 300 days history, no exclusion span) on the event "
        "date and no data artifact inside the pattern window; counts from "
        f"{start:%Y-%m-%d}.",
        "",
        "## Event counts",
        "",
    ]
    t = pd.DataFrame(index=pd.Index(PATTERNS, name="pattern"))
    for st in STAGES:
        t[f"{st} (all)"] = ev[ev["stage"] == st].groupby("pattern").size()
        t[f"{st} (eligible)"] = ev[elig & (ev["stage"] == st)].groupby("pattern").size()
    L += md_table(_fmt(t.fillna(0).astype(int))) + [""]
    trig = ev[elig & (ev["stage"] == "triggered")]
    n_pre = (trig["date"] < HOLDOUT).sum()
    L += [
        f"Eligible triggered events before 2024: {n_pre:,}; from 2024 (locked holdout, "
        f"detections only, no outcomes looked at): {len(trig) - n_pre:,}.",
        "",
        "Eligible triggered events at the other liquidity thresholds (elig bits):",
        "",
    ]
    bits = json.loads((resolve(cfg["paths"]["features"]) / "meta.json").read_text())["elig_bits"]
    base = ev[~ev["data_flag"] & (ev["date"] >= start) & (ev["stage"] == "triggered")]
    tb = pd.DataFrame(index=pd.Index(PATTERNS, name="pattern"))
    for i, thr in enumerate(bits):
        tb[f"${thr / 1e6:.0f}M"] = (
            base[(base["elig"].to_numpy() >> i & 1) == 1].groupby("pattern").size()
        )
    L += md_table(_fmt(tb.fillna(0).astype(int))) + [""]

    L += ["## Triggered events per year (eligible)", ""]
    py = trig.groupby([trig["date"].dt.year.rename("year"), "pattern"]).size().unstack(fill_value=0)
    py = py.reindex(columns=list(PATTERNS), fill_value=0)
    fs_rows = resolve(cfg["paths"]["features"]) / "rows.parquet"
    fsr = pd.read_parquet(fs_rows, columns=["date", "elig"])
    ed = fsr[(fsr["elig"] & 1) == 1].groupby(fsr["date"].dt.year).size()
    py2 = py.copy()
    for p in PATTERNS:
        py2[f"{p} per 10k elig. days"] = (1e4 * py[p] / ed.reindex(py.index)).round(1)
    L += md_table(_fmt(py2)) + [""]
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
    for p in PATTERNS:
        ax[0].plot(py.index, py[p], label=p)
        ax[1].plot(py.index, py2[f"{p} per 10k elig. days"], label=p)
    ax[0].set_title("eligible triggered events per year")
    ax[1].set_title("per 10,000 eligible stock-days")
    for a in ax:
        a.axvline(2024, color="#999", ls="--", lw=0.8)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "events_per_year.png", dpi=120)
    plt.close(fig)
    L += ["![events per year](figures/B2_patterns/events_per_year.png)", ""]

    L += ["## Triggered events per liquidity bucket (50-day dollar volume, eligible)", ""]
    lb = pd.cut(trig["dollar_vol50"], LIQ_BINS, labels=LIQ_NAMES, right=False)
    tl = trig.groupby([lb.rename("bucket"), "pattern"], observed=False).size().unstack(fill_value=0)
    L += md_table(_fmt(tl.reindex(columns=list(PATTERNS)))) + [""]

    L += ["## Geometry of eligible triggered events (p10 / median / p90)", ""]
    geo = {
        "cup_handle": [
            "prior_rise",
            "lip_vs_high",
            "cup_depth",
            "cup_low_vix_pct",
            "cup_days",
            "lip_diff",
            "bottom_frac",
            "cup_r2",
            "handle_days",
            "handle_cup_ratio",
            "handle_depth",
            "handle_pos",
            "handle_slope",
            "handle_vol_ratio",
            "vol_ratio_t",
        ],
        "hs_top": [
            "prior_trend",
            "head_vs_extreme",
            "head_prominence",
            "head_prominence_atr",
            "head_prominence_close_atr",
            "head_height",
            "ls_height",
            "rs_height",
            "shoulder_diff",
            "shoulder_diff_hh",
            "time_sym",
            "neck_slope",
            "span_days",
            "scale",
            "vol_ratio_t",
        ],
    }
    geo["inv_hs"] = geo["hs_top"]
    for p in PATTERNS:
        q = trig[trig["pattern"] == p][geo[p]].quantile([0.1, 0.5, 0.9]).T
        q.columns = ["p10", "median", "p90"]
        q.index.name = p
        L += md_table(q.map(lambda v: f"{v:.3g}")) + [""]
    risk = trig["trigger_price"] - trig["stop"]
    rr = trig.assign(rr=(trig["target"] - trig["trigger_price"]) / risk)
    q = rr.groupby("pattern")["rr"].quantile([0.1, 0.5, 0.9]).unstack()
    q.index.name = "textbook reward/risk vs trigger price"
    L += md_table(q.map(lambda v: f"{v:.2f}")) + [""]

    L += ["## Stage funnel (eligible forming events before 2024, same pattern instance)", ""]
    pre = ev[(ev["date"] >= start) & (ev["date"] < HOLDOUT) & ~ev["data_flag"]]
    fo = pre[(pre["stage"] == "forming") & ((pre["elig"] & 1) == 1)]
    trig_row = pre[pre["stage"] == "triggered"].set_index("pattern_id")["row"]
    fu = pd.DataFrame(index=pd.Index(PATTERNS, name="pattern"))
    for p in PATTERNS:
        f = fo[fo["pattern"] == p]
        lead = trig_row.reindex(f["pattern_id"]).to_numpy() - f["row"].to_numpy()
        hit = np.isfinite(lead)
        fu.loc[p, "forming"] = f"{len(f):,}"
        fu.loc[p, "→ triggered"] = f"{100 * hit.mean():.1f}%" if len(f) else "–"
        fu.loc[p, "days to trigger (median)"] = f"{np.median(lead[hit]):.0f}" if hit.any() else "–"
    L += md_table(fu) + [
        "",
        "Descriptive only (pre-2024). The conversion labels with a fixed horizon N come in B3.",
        "",
    ]

    L += ["## Loose (labeling) thresholds", ""]
    lt = loose[((loose["elig"] & 1) == 1) & ~loose["data_flag"] & (loose["date"] >= start)]
    L += [
        f"Eligible loose events: {len(lt):,} (default run: {int(elig.sum()):,}). "
        "Share failing at least one default gate, and the most frequent failing gates:",
        "",
    ]
    rows = []
    for p in PATTERNS:
        for st in STAGES:
            c = lt[(lt["pattern"] == p) & (lt["stage"] == st)]
            fails = c["failed_default"].str.split(";").explode()
            top = fails[fails != ""].value_counts().head(4)
            rows.append(
                {
                    "pattern / stage": f"{p} / {st}",
                    "events": f"{len(c):,}",
                    "fail ≥ 1 default gate": f"{100 * (c['failed_default'] != '').mean():.0f}%",
                    "top failing gates": ", ".join(
                        f"{k} {100 * v / len(c):.0f}%" for k, v in top.items()
                    ),
                }
            )
    L += md_table(pd.DataFrame(rows).set_index("pattern / stage")) + [""]

    sp = resolve(cfg["paths"]["out_dir"]) / "label_sample.parquet"
    if sp.exists():
        s = pd.read_parquet(sp)
        L += ["## Hand-labeling sample", ""]
        L += [
            f"{len(s)} events (`artifacts/patterns/label_sample.parquet`), drawn with the loose "
            f"thresholds from eligible events dated {cfg['build']['report_start']} → "
            f"{cfg['sample']['end']} (never from the holdout), one per pattern instance, "
            f"stratified by pattern × stage, {100 * cfg['sample']['borderline_share']:.0f}% "
            "borderline (fail ≥ 1 default gate):",
            "",
        ]
        ct = s.groupby(["pattern", "stage", "borderline"]).size().unstack(fill_value=0)
        ct.columns = ["passes default", "borderline"]
        ct.index = [f"{a} / {b}" for a, b in ct.index]
        ct.index.name = "pattern / stage"
        L += md_table(ct) + [""]
        lab = load_labels(resolve(cfg["paths"]["labels"]))
        L += [f"Labels so far: {len(lab)} (`data/labels/human_checks.csv`).", ""]

    gd = resolve(cfg["paths"]["gallery"])
    if (gd / "index.html").exists():
        L += [
            "## Gallery",
            "",
            "50 random eligible triggered detections per pattern (trigger dates up to "
            f"{cfg['gallery']['end']}, so the {cfg['gallery']['after_days']}-day 'after' window "
            "stays out of the holdout): "
            + ", ".join(f"[{p}](gallery/{p}.html)" for p in PATTERNS)
            + ".",
            "",
        ]
    L += [
        "## Tests",
        "",
        "- `tests/test_patterns.py`: pivots (confirmation timing, alternation), synthetic "
        "cup-and-handle / H&S top / inverse H&S with known geometry and trigger day, stage "
        "order, reason codes in sync with the reference sheet.",
        "- `tests/test_patterns_truncation.py`: events dated ≤ t from data cut at t equal the "
        "full-history events (every column, exact), default and loose thresholds, synthetic "
        "and random real tickers.",
        "",
        "## Limitations",
        "",
        "- Survivorship bias: the universe is today's listings; delisted stocks (many failed "
        "breakouts among them) are missing.",
        "- yfinance data quality: adjusted prices, occasional bad prints; patterns touching a "
        "known artifact are flagged (`data_flag`) and excluded from counts, gallery and sample.",
        "- Pattern definitions are subjective; the thresholds are a starting point for the "
        "hand-label review.",
        "",
    ]
    resolve(cfg["paths"]["report"]).write_text("\n".join(L))
    log.info("wrote %s", cfg["paths"]["report"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", nargs="+", default=["build", "sample", "gallery", "report"])
    ap.add_argument("--tickers", nargs="+")
    ap.add_argument("--out-dir")
    ap.add_argument("--workers", type=int)
    args = ap.parse_args()
    cfg = load_config("detector")
    if args.out_dir:
        cfg["paths"]["out_dir"] = args.out_dir
    setup_logging(resolve(cfg["paths"]["log_file"]))
    if "build" in args.steps:
        build(cfg, project_root(), args.workers, args.tickers)
    if "sample" in args.steps:
        make_sample(cfg)
    if "gallery" in args.steps:
        make_gallery(cfg)
    if "report" in args.steps:
        write_report(cfg)


if __name__ == "__main__":
    main()
