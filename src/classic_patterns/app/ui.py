"""Screener UI: streamlit run src/classic_patterns/app/ui.py

Shows the long setups scored by ``scripts/screen.py`` (``artifacts/screener/<date>.parquet``):
filterable tables of triggered setups (probabilities of the chosen model) and forming setups
(chance of triggering), the market regime, and a chart per setup with the trigger, stop and target.
It shows pattern setups with estimated probabilities; you decide.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from classic_patterns.app import charts
from classic_patterns.app.display import TARGET_NOTE, with_distances
from classic_patterns.utils.config import resolve

BANNER = (
    "**Shows pattern setups with estimated probabilities; you decide.** Do not take every pattern "
    "it finds — use your own judgement. Research / education project — **not investment or "
    "financial advice.**"
)
MODELS = {"student (Chronos-2-Small, ONNX)": "student", "LightGBM (hand-crafted features)": "lgbm"}


@st.cache_data
def load_day(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


@st.cache_data
def load_prices(ticker: str) -> pd.DataFrame:
    return charts.load_prices(ticker)


def main() -> None:
    st.set_page_config(page_title="classic-patterns screener", layout="wide")
    st.warning(BANNER)
    files = sorted(resolve("artifacts/screener").glob("*.parquet"))
    if not files:
        st.info("No screener output yet: run `python scripts/screen.py --refresh`.")
        st.stop()
    side = st.sidebar
    day = side.selectbox("Session", [f.stem for f in reversed(files)])
    df = load_day(str(resolve("artifacts/screener") / f"{day}.parquet"))
    if df.empty:
        st.info(f"No setups on {day}.")
        st.stop()
    m = MODELS[side.radio("Model", list(MODELS))]
    pats = side.multiselect(
        "Pattern", sorted(df["pattern"].unique()), sorted(df["pattern"].unique())
    )
    stages = side.multiselect("Stage", ["triggered", "forming"], ["triggered", "forming"])
    pmax = side.slider("Max P(stop first), triggered setups", 0.0, 1.0, 1.0, 0.05)
    f = df[df["pattern"].isin(pats) & df["stage"].isin(stages)
           & ((df["stage"] == "forming") | (df[f"{m}_p_stop"] <= pmax))].copy()  # fmt: skip

    r0 = df.iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Setups", f"{len(f)} of {len(df)}")
    c2.metric("VIX", f"{r0['vix']:.1f}")
    c3.metric("S&P 500 vs 200-day average", "above" if r0["gspc_above200"] else "below")

    show = with_distances(f).rename(columns={
        f"{m}_p_target": "P(target)", f"{m}_p_stop": "P(stop first)",
        f"{m}_p_timeout": "P(neither, 60d)", f"{m}_exp_r": "exp. R", "risk_pct": "risk %",
        "target": "target (reference)", "target_pct": "target away %",
        "to_trigger_pct": "to trigger %", "student_p_trigger20": "P(trigger in 20d)",
    }).sort_values("ticker")  # fmt: skip
    triggered = show[show["stage"] == "triggered"]
    forming = show[show["stage"] == "forming"]
    st.info(TARGET_NOTE)

    st.subheader("Triggered")
    st.caption(
        "Probabilities for the pattern's own plan: entry at the next open, the pattern's stop and "
        "target, at most 60 trading days; calibrated on the last ~2 years of training data. "
        "Expected R uses training-period averages. Distances are from the close. Sorted by "
        "ticker — not a ranking of trades."
    )
    st.dataframe(triggered[["ticker", "pattern", "close", "stop", "risk %", "P(stop first)",
                            "P(neither, 60d)", "P(target)", "exp. R", "target (reference)",
                            "target away %"]].round(3), hide_index=True,
                 width="stretch")  # fmt: skip

    st.subheader("Forming")
    st.caption(
        "Not triggered yet, so no trade estimates. Risk is measured from the trigger level. "
        "P(trigger in 20d) comes from the student model, whichever model is selected."
    )
    st.dataframe(forming[["ticker", "pattern", "close", "trigger", "to trigger %", "stop", "risk %",
                          "P(trigger in 20d)"]].round(3), hide_index=True,
                 width="stretch")  # fmt: skip

    if len(f):
        pick = st.selectbox("Chart", f["ticker"] + " — " + f["pattern"] + " (" + f["stage"] + ")")
        row = f.iloc[list(f["ticker"] + " — " + f["pattern"] + " (" + f["stage"] + ")").index(pick)]
        days = st.slider("Days shown", 60, 500, 250, 10)
        st.plotly_chart(charts.chart(row, load_prices(row["ticker"]), days), width="stretch")
        if row["stage"] == "triggered":
            est = (
                f"Model: {m}. P(stop first) {row[f'{m}_p_stop']:.2f}, P(neither, 60d) "
                f"{row[f'{m}_p_timeout']:.2f}, P(target) {row[f'{m}_p_target']:.2f}."
            )
        else:
            est = f"Student model: P(trigger in 20d) {row['student_p_trigger20']:.2f}."
        st.caption(f"{est} You decide; this is not a recommendation.")


main()
