"""Hand-labeling: reason codes, the labeling sample and the human-checks CSV.

Reason codes must match the tables of the reference sheets (checked by tests when the sheets are
present).
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from classic_patterns.patterns.common import PATTERNS, STAGES

LABELS = ("valid", "invalid", "unsure")
# who set a label: "human" (labeling page) or "llm" (blind chart review);
# rows without the column predate it and are human
LABELERS = ("human", "llm")
REASONS_CUP = (
    "v_shape", "cup_too_shallow", "cup_too_deep", "cup_too_short", "cup_too_long", "lips_uneven",
    "handle_too_deep", "handle_too_low", "handle_too_short", "handle_too_long", "handle_rising",
)  # fmt: skip
REASONS_HS = (
    "head_not_prominent", "shoulders_uneven", "shoulders_asymmetric_time", "shoulders_too_high",
    "shoulders_too_low", "neckline_too_steep", "not_a_reversal",
)  # fmt: skip
REASONS_ANY = ("no_prior_trend", "wrong_pivots", "noisy", "data_issue", "other")
# consolidation_breakouts.md, in the sheet's order (the shared codes are part of each list there)
REASONS_CB = (
    "no_prior_trend", "resistance_not_near_highs", "resistance_not_flat", "too_few_touches",
    "too_short", "too_long", "too_deep", "too_small", "lows_falling", "not_contained", "noisy",
    "wrong_pivots", "data_issue", "other",
)  # fmt: skip
REASONS_AT = (
    "top_not_flat", "lows_not_rising", "too_few_touches", "too_short", "too_long", "too_small",
    "breakout_at_apex", "not_contained", "noisy", "wrong_pivots", "no_prior_trend", "data_issue",
    "other",
)  # fmt: skip
REFERENCE = {  # reference sheet per pattern (shown in the labeling app)
    "consolidation_breakout": "docs/labeling/consolidation_breakouts.pdf",
    "ascending_triangle": "docs/labeling/consolidation_breakouts.pdf",
}
CSV_COLS = [
    "sample_id", "event_id", "ticker", "pattern", "stage", "date", "label", "reasons", "note",
    "labeled_at", "sample_file", "labeler",
]  # fmt: skip


def reason_codes(pattern: str) -> tuple[str, ...]:
    if pattern == "consolidation_breakout":
        return REASONS_CB
    if pattern == "ascending_triangle":
        return REASONS_AT
    return (REASONS_CUP if pattern == "cup_handle" else REASONS_HS) + REASONS_ANY


def reference_sheet(pattern: str) -> str:
    return REFERENCE.get(pattern, "docs/labeling/pattern_reference.pdf")


def draw_sample(loose: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Stratified labeling sample from the loose-threshold events: per pattern x stage,
    ``per_cell`` events, ``borderline_share`` of them failing at least one default gate.
    Only eligible (default liquidity rule), artifact-free events dated <= ``end``; one event per
    pattern instance and at most one per ticker and cell. Returned in random order."""
    sc = cfg["sample"]
    rng = np.random.default_rng(sc["seed"])
    ev = _pool(loose, cfg)
    used: set[str] = set()
    picks = []
    for pat in PATTERNS:
        for stage in STAGES:
            cell = ev[(ev["pattern"] == pat) & (ev["stage"] == stage)]
            n_border = int(round(sc["per_cell"] * sc["borderline_share"]))
            want = {True: n_border, False: sc["per_cell"] - n_border}
            for flag in (False, True):
                pool = cell[cell["borderline"] == flag]
                pool = pool.iloc[rng.permutation(len(pool))]
                tickers: set[str] = set()
                got = 0
                for r in pool.itertuples():
                    if got >= want[flag]:
                        break
                    if r.pattern_id in used or r.ticker in tickers:
                        continue
                    used.add(r.pattern_id)
                    tickers.add(r.ticker)
                    picks.append(r.Index)
                    got += 1
    out = ev.loc[picks].iloc[rng.permutation(len(picks))].reset_index(drop=True)
    out.insert(0, "sample_id", np.arange(1, len(out) + 1))
    return out.assign(top_up=False, top_up_id=-1)


def _pool(loose: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    sc = cfg["sample"]
    ev = loose[
        (loose["date"] <= sc["end"])
        & (loose["date"] >= cfg["build"]["report_start"])
        & ((loose["elig"] & 1) == 1)
        & ~loose["data_flag"]
    ]
    return ev.assign(borderline=ev["failed_default"].str.len() > 0)


def top_up_sample(
    sample: pd.DataFrame, loose: pd.DataFrame, cfg: dict, labels: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Append the events asked for in ``sample.top_up`` (pattern, stage, n; optional borderline and
    a pandas ``query`` on the event columns) after the existing ones, so labeled events keep their
    sample_id. Idempotent: counts the rows already added per entry (``top_up_id`` = entry index).
    Same eligibility as ``draw_sample``; new pattern instances and tickers only.

    An entry with ``seed_from`` ({pattern, reason, window_days}) instead takes, for each human
    label of that pattern with that reason code, the event of ``pattern`` on the same ticker
    whose date is closest to the labeled event's (within ``window_days`` calendar days; stage
    filter optional)."""
    ev = _pool(loose, cfg)
    if "top_up_id" not in sample:  # samples from before top_up_id: all top-ups are entry 0
        sample = sample.assign(top_up_id=np.where(sample["top_up"], 0, -1))
    used = set(sample["pattern_id"])
    out = [sample]
    nxt = int(sample["sample_id"].max()) + 1
    for i, u in enumerate(cfg["sample"].get("top_up") or []):
        need = u["n"] - int((sample["top_up_id"] == i).sum())
        if need <= 0:
            continue
        if u.get("seed_from"):
            add = _seed_events(ev, sample, labels, u, need, used)
            add = add.assign(top_up=True, top_up_id=i)
            add.insert(0, "sample_id", np.arange(nxt, nxt + len(add)))
            nxt += len(add)
            out.append(add)
            continue
        cell = ev[(ev["pattern"] == u["pattern"]) & (ev["stage"] == u["stage"])]
        if u.get("borderline") is not None:
            cell = cell[cell["borderline"] == u["borderline"]]
        if u.get("query"):
            cell = cell.query(u["query"])
        pool = cell[~cell["pattern_id"].isin(used)]
        rng = np.random.default_rng([cfg["sample"]["seed"], i])
        tickers = set(sample.loc[(sample["pattern"] == u["pattern"]), "ticker"])
        picks = []
        for r in pool.iloc[rng.permutation(len(pool))].itertuples():
            if len(picks) >= need:
                break
            if r.pattern_id in used or r.ticker in tickers:
                continue
            used.add(r.pattern_id)
            tickers.add(r.ticker)
            picks.append(r.Index)
        add = pool.loc[picks].reset_index(drop=True).assign(top_up=True, top_up_id=i)
        add.insert(0, "sample_id", np.arange(nxt, nxt + len(add)))
        nxt += len(add)
        out.append(add)
    res = pd.concat(out, ignore_index=True)
    if "detected" in res:  # rows added here come from the current detectors
        res["detected"] = res["detected"].astype("boolean").fillna(True).astype(bool)
    return res


def refresh_sample(old: pd.DataFrame, loose: pd.DataFrame) -> pd.DataFrame:
    """The same events in the same order, with columns and ``borderline`` taken from a rebuilt
    ``loose`` table. Labels are keyed by event_id, so a sample with labels is never redrawn.
    An event the rebuilt detectors no longer find keeps its previous row (the chart that was
    labeled) and gets ``detected = False``."""
    new = loose.set_index("event_id").reindex(old["event_id"])
    missing = new["pattern"].isna().to_numpy()
    out = new.reset_index()[list(loose.columns)]
    if missing.any():
        prev = old.reset_index(drop=True).reindex(columns=out.columns)
        out = out.astype(object)
        out.loc[missing] = prev.loc[missing].to_numpy()
        out = out.infer_objects()
        for c in loose.columns:  # keep the loose table's dtypes where possible
            try:
                out[c] = out[c].astype(loose[c].dtype)
            except (TypeError, ValueError):
                pass
    out["detected"] = ~missing
    out = out.assign(borderline=out["failed_default"].str.len() > 0)
    out.insert(0, "sample_id", old["sample_id"].to_numpy())
    top = old["top_up"].to_numpy() if "top_up" in old else np.zeros(len(old), dtype=bool)
    top_id = old["top_up_id"].to_numpy() if "top_up_id" in old else np.where(top, 0, -1)
    return out.assign(top_up=top, top_up_id=top_id)


def load_labels(path: Path) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame(columns=CSV_COLS)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "labeler" not in df:
        df["labeler"] = "human"
    return df


def save_label(path: Path, row: dict) -> pd.DataFrame:
    """Upsert one label (keyed by event_id) and rewrite the CSV atomically. ``labeler`` defaults
    to "human"."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = load_labels(path)
    row = {k: str(row.get(k, "")) for k in CSV_COLS}
    row["labeler"] = row["labeler"] or "human"
    if row["labeler"] not in LABELERS:
        raise ValueError(f"unknown labeler {row['labeler']!r}")
    row["labeled_at"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    df = df[df["event_id"] != row["event_id"]]
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df["_n"] = pd.to_numeric(df["sample_id"], errors="coerce")
    df = df.sort_values("_n").drop(columns="_n")
    tmp = path.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)
    return df


def _seed_events(ev, sample, labels, u: dict, need: int, used: set) -> pd.DataFrame:
    """Events of ``u['pattern']`` near human-labeled events with a given reason code."""
    sf = u["seed_from"]
    if labels is None or not len(labels):
        return ev.iloc[0:0].reset_index(drop=True)
    lab = labels[(labels["labeler"] == "human") & (labels["pattern"] == sf["pattern"])]
    lab = lab[lab["reasons"].fillna("").str.split(";").apply(lambda r: sf["reason"] in r)]
    cand = ev[ev["pattern"] == u["pattern"]]
    if u.get("stage"):
        cand = cand[cand["stage"] == u["stage"]]
    win = pd.Timedelta(days=sf.get("window_days", 60))
    picks = []
    for r in lab.sort_values("sample_id").itertuples():
        if len(picks) >= need:
            break
        c = cand[(cand["ticker"] == r.ticker) & ~cand["pattern_id"].isin(used)]
        if c.empty:
            continue
        dist = (c["date"] - pd.Timestamp(r.date)).abs()
        c = c[dist <= win]
        if c.empty:
            continue
        best = c.loc[(c["date"] - pd.Timestamp(r.date)).abs().idxmin()]
        used.add(best["pattern_id"])
        picks.append(best.name)
    return ev.loc[picks].reset_index(drop=True)
