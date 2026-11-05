"""Interactive pattern charts (plotly) for the gallery and the hand-labeling page.

Colors are neutral on purpose: nothing on the chart says whether a measured value passes a rule.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from classic_patterns.features.indicators import sma
from classic_patterns.patterns.common import CUP_PATTERNS

PRE = {"up": "#3b4a5a", "down": "#a9b4c0"}  # up to the detection day
POST = {"up": "#2f6fb0", "down": "#9cc3e8"}  # after it (gallery only)
LINE = "#c27c0e"
MARK = "#6b3fa0"
LEVEL = {"stop": "#555555", "target": "#555555"}
MA_COLORS = ("#7a9a3a", "#b0566e")
# price fields of an event, rescaled with the bars on blind charts
PRICE_FIELDS = ("trigger_price", "trigger_slope", "handle_low", "resistance", "stop", "target")

PIVOT_NAMES = {
    "cup_handle": ("left lip", "cup low", "right lip", "handle low", None),
    "hs_top": ("LS", "T1", "head", "T2", "RS"),
    "inv_hs": ("LS", "T1", "head", "T2", "RS"),
    "deep_cup": ("left lip", "cup low", "right lip", "handle low", None),
    "consolidation_breakout": ("first touch", "base low", "last touch", "last swing low", None),
    "ascending_triangle": ("first top", "first low", "last top", "support low", None),
}
# label position per pivot number, where the default (peaks above, troughs below) overlaps
LABEL_POS = {
    "cup_handle": {3: "top left"},
    "deep_cup": {3: "top left"},
    "consolidation_breakout": {3: "top left"},
    "ascending_triangle": {3: "top left"},
}
CONS = ("consolidation_breakout", "ascending_triangle")
SUBTYPE_TEXT = ("flat base", "ascending", "tightening", "flag")
STAGE_TEXT = {
    "forming": "forming (handle / right shoulder in place, price near the trigger)",
    "triggered": "triggered (close through the trigger level)",
}
PATTERN_TEXT = {
    "cup_handle": "Cup-and-handle",
    "hs_top": "Head-and-shoulders top",
    "inv_hs": "Inverse head-and-shoulders",
    "deep_cup": "Deep cup (correction and recovery to the old high)",
    "consolidation_breakout": "Consolidation breakout at highs",
    "ascending_triangle": "Ascending triangle",
}


def _pct(x, d=1):
    return "–" if not np.isfinite(x) else f"{100 * x:.{d}f}%"


def _num(x, d=2):
    return "–" if not np.isfinite(x) else f"{x:.{d}f}"


def measured_values(e) -> list[tuple[str, str]]:
    """(name, formatted value) for the measured geometry of one event, in reading order."""
    g = lambda k: float(e[k]) if e[k] is not None else np.nan  # noqa: E731
    if e["pattern"] in CUP_PATTERNS:
        rows = [
            ("Prior uptrend into the left lip", _pct(g("prior_rise"), 0)),
            ("Left lip vs highest high of the 6 months before", _pct(g("lip_vs_high"))),
            ("Cup depth", _pct(g("cup_depth"))),
            ("VIX 1-year percentile near the cup low", _pct(g("cup_low_vix_pct"), 0)),
            ("Cup duration", f"{g('cup_days'):.0f} days ({g('cup_days') / 5:.1f} weeks)"),
            ("Right lip vs left lip", _pct(g("lip_diff"))),
            ("Right lip: share of the cup depth regained", _pct(g("lip_recovery"), 0)),
            ("Cup low vs start of the prior uptrend", _pct(g("cup_low_vs_trend"), 0)),
            ("Lips on spike / gap bars", _num(g("spike_lips"), 0)),
            ("Share of cup closes in its lower third", _pct(g("bottom_frac"), 0)),
            ("Cup quadratic-fit R²", _num(g("cup_r2"))),
            ("Cup low position (0 = left lip, 1 = right lip)", _num(g("cup_low_pos"))),
            ("Handle length", f"{g('handle_days'):.0f} days"),
            ("Handle length / cup length", _num(g("handle_cup_ratio"))),
            ("Handle depth (from pivot)", _pct(g("handle_depth"))),
            ("Handle low, share of cup depth retraced", _pct(g("handle_pos"), 0)),
            ("Handle slope (closes, per day)", _pct(g("handle_slope"), 2)),
            ("Handle volume / 50-day average before it", _num(g("handle_vol_ratio"))),
            ("Volume on this day / 50-day average", _num(g("vol_ratio_t"))),
            ("Pivot (trigger price)", _num(g("trigger_price"))),
        ]
    elif e["pattern"] in CONS:
        sub = g("subtype_code")
        rows = [
            ("Prior rise into the first touch", _pct(g("prior_rise"), 0)),
            ("Prior change (vs highest high before)", _pct(g("prior_drop"), 0)),
            ("Resistance vs 6-month high before the base", _pct(g("res_vs_high"))),
            ("Resistance touches", f"{g('touches'):.0f}"),
            ("Flatness (touch levels, ATR14)", _num(g("flatness_atr"))),
            ("Duration (first touch → this day)", f"{g('duration_days'):.0f} days"),
            ("Depth", _pct(g("base_depth"))),
            ("Height (ATR14)", _num(g("height_atr"), 1)),
            ("Swing lows inside", f"{g('n_lows'):.0f}"),
            ("Lows: last − first (ATR14)", _num(g("lows_trend_atr"), 1)),
            ("Lows: smallest rise (ATR14)", _num(g("lows_rise_min_atr"), 1)),
            ("Closes outside the range before", f"{g('closes_out'):.0f}"),
            ("Tightness (last 10 days / height)", _num(g("tightness"))),
            ("Sub-type", SUBTYPE_TEXT[int(sub)] if np.isfinite(sub) else "–"),
            ("Volume in the base / 50-day average before", _num(g("base_vol_ratio"))),
            ("Volume on this day / 50-day average", _num(g("vol_ratio_t"))),
            ("Resistance", _num(g("resistance"))),
        ]
        if e["pattern"] == "ascending_triangle":
            bd = g("breakout_dir")
            rows += [
                ("Support slope (ATR14 per day)", _num(g("support_slope_atr"), 3)),
                ("Breakout position (0 = first touch, 1 = apex)", _num(g("breakout_pos"))),
                (
                    "Breakout direction",
                    "–" if not np.isfinite(bd) else ("up" if bd > 0 else "down"),
                ),
            ]
        if e["stage"] == "triggered":
            rows.append(
                ("Opening gap beyond the line on the trigger day (ATR)", _num(g("gap_atr_t")))
            )
    else:
        rows = [
            ("Prior trend into the left shoulder", _pct(g("prior_trend"), 0)),
            ("Head beyond the 6-month extreme before it", _pct(g("head_vs_extreme"))),
            ("Head beyond the more extreme shoulder", _pct(g("head_prominence"))),
            ("  … in ATR14 at the head", _num(g("head_prominence_atr"), 1)),
            ("  … on closes, in ATR14", _num(g("head_prominence_close_atr"), 1)),
            ("Head distance from the neckline", _pct(g("head_height"))),
            ("Left shoulder height / head height", _num(g("ls_height"))),
            ("Right shoulder height / head height", _num(g("rs_height"))),
            ("Shoulder price difference", _pct(g("shoulder_diff"))),
            ("  … / head height", _num(g("shoulder_diff_hh"))),
            ("Time symmetry (right side / left side)", _num(g("time_sym"))),
            ("Neckline change LS→RS / head height", _num(g("neck_slope"))),
            ("Left shoulder → head", f"{g('left_days'):.0f} days"),
            ("Left shoulder → right shoulder", "–" if not np.isfinite(g("span_days"))
             else f"{g('span_days'):.0f} days"),
            ("Volume at LS / head / RS (vs 50-day avg)",
             f"{_num(g('vol_ls'))} / {_num(g('vol_h'))} / {_num(g('vol_rs'))}"),
            ("Volume on this day / 50-day average", _num(g("vol_ratio_t"))),
            ("Neckline on this day (trigger price)", _num(g("trigger_price"))),
            ("Zigzag scale (× ATR)", _num(g("scale"), 1)),
            ("Smallest leg between pivots (ATR14)", _num(g("min_leg_atr"), 1)),
            ("Fewest bars between neighbouring pivots", _num(g("min_spacing"), 0)),
            ("Pivots on spike / gap bars", _num(g("spike_pivots"), 0)),
        ]  # fmt: skip
    return rows


def pattern_figure(
    bars: pd.DataFrame,
    e,
    before: int = 120,
    after: int = 0,
    log_y: bool = False,
    show_levels: bool = False,
    height: int = 620,
    blind: bool = False,
    mas: tuple[int, ...] = (),
    benchmark: pd.DataFrame | None = None,
    level_days: int = 15,
) -> go.Figure:
    """Chart of one event. ``bars``: the ticker's rows (date, open, high, low, close, volume) in
    the same order as the detector input. ``after`` > 0 shows that many days past the event
    (gallery); 0 hides everything after the detection day (labeling).

    ``blind``: x axis in trading days relative to the detection day ("t-12"), prices rebased to
    100 at its close, volume as a multiple of its 50-day average up to that day. ``mas``: moving
    averages of the close. ``benchmark`` (date, close): third panel with its 200-day average,
    aligned on the event's dates. With ``after`` = 0, stop / target span the last ``level_days``
    bars."""
    t = int(e["row"])
    a = max(0, int(e["start_row"]) - before)
    z = min(len(bars), t + 1 + after)
    if blind:
        bars, e = _rebased(bars, e, t)
    w = bars.iloc[a:z]
    x = _rel(np.arange(a, z), t) if blind else w["date"].dt.strftime("%Y-%m-%d").to_numpy()
    pre = np.arange(a, z) <= t
    n_rows = 2 if benchmark is None else 3
    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.78, 0.22] if n_rows == 2 else [0.6, 0.15, 0.25],
        vertical_spacing=0.03,
    )
    for part, col, name in ((pre, PRE, "price"), (~pre, POST, "after detection")):
        if not part.any():
            continue
        fig.add_trace(
            go.Candlestick(
                x=x[part],
                open=w["open"].to_numpy()[part],
                high=w["high"].to_numpy()[part],
                low=w["low"].to_numpy()[part],
                close=w["close"].to_numpy()[part],
                increasing=dict(line=dict(color=col["up"], width=1), fillcolor=col["up"]),
                decreasing=dict(line=dict(color=col["down"], width=1), fillcolor=col["down"]),
                name=name,
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        up = (w["close"] >= w["open"]).to_numpy()[part]
        fig.add_trace(
            go.Bar(
                x=x[part],
                y=w["volume"].to_numpy()[part],
                marker_color=np.where(up, col["up"], col["down"]),
                showlegend=False,
                name="volume",
            ),
            row=2,
            col=1,
        )

    def d(r):
        if blind:
            return _rel(np.array([int(r)]), t)[0]
        return bars["date"].iloc[int(r)].strftime("%Y-%m-%d")

    close = bars["close"].to_numpy()
    for n, col in zip(mas, MA_COLORS * len(mas), strict=False):
        fig.add_trace(
            go.Scatter(x=x, y=sma(close, n)[a:z], mode="lines", line=dict(color=col, width=1),
                       name=f"{n}-day avg", showlegend=False, hoverinfo="skip"),
            row=1, col=1,
        )  # fmt: skip
    if benchmark is not None:
        bm = benchmark.sort_values("date").set_index("date")["close"]
        bm = pd.DataFrame({"c": bm, "m": sma(bm.to_numpy(), 200)}, index=bm.index)
        bm = bm.reindex(w["date"], method="ffill")
        if blind:
            bm = bm * 100 / bm["c"].iloc[t - a]
        for k, dash in (("c", "solid"), ("m", "dot")):
            fig.add_trace(
                go.Scatter(x=x, y=bm[k].to_numpy(), mode="lines", showlegend=False,
                           line=dict(color="#3b4a5a", width=1.2 if k == "c" else 1, dash=dash),
                           name="S&P 500" if k == "c" else "S&P 500 200-day avg"),
                row=3, col=1,
            )  # fmt: skip
        fig.update_yaxes(title_text="S&P 500", title_font=dict(size=10), row=3, col=1)

    names = PIVOT_NAMES[e["pattern"]]
    peak_hi = e["pattern"] != "inv_hs"
    pts = []
    for k, nm in enumerate(names, start=1):
        r = int(e[f"p{k}_row"])
        if nm is None or r < 0:
            continue
        is_peak = (k % 2 == 1) == peak_hi
        if e["pattern"] in CUP_PATTERNS or e["pattern"] in CONS:
            is_peak = k in (1, 3)
        y = bars["high"].iloc[r] if is_peak else bars["low"].iloc[r]
        pos = LABEL_POS.get(e["pattern"], {}).get(k)
        pts.append((d(r), y, nm, is_peak, pos))
    fig.add_trace(
        go.Scatter(
            x=[p[0] for p in pts],
            y=[p[1] for p in pts],
            mode="lines+markers+text",
            text=[p[2] for p in pts],
            textposition=[p[4] or ("top center" if p[3] else "bottom center") for p in pts],
            line=dict(color=MARK, width=1, dash="dot"),
            marker=dict(color=MARK, size=8),
            showlegend=False,
            name="pivots",
        ),
        row=1,
        col=1,
    )
    t_end = min(len(bars) - 1, t + after)
    if e["pattern"] in CUP_PATTERNS:
        r3 = int(e["p3_row"])
        piv = float(e["trigger_price"])
        _hline(fig, d(r3), d(t_end), piv, "pivot", LINE, pos="bottom left")
        _hline(
            fig, d(r3), d(t), float(e["handle_low"]), "", LINE, dash="dot"
        )  # marker has the name
    elif e["pattern"] in CONS:
        _hline(fig, d(int(e["p1_row"])), d(t_end), float(e["resistance"]), "resistance", LINE,
               pos="bottom left")  # fmt: skip
        if e["pattern"] == "ascending_triangle" and int(e["p4_row"]) > int(e["p2_row"]) >= 0:
            r2, r4 = int(e["p2_row"]), int(e["p4_row"])
            y2, y4 = float(bars["low"].iloc[r2]), float(bars["low"].iloc[r4])
            m = (y4 - y2) / (r4 - r2)
            fig.add_trace(
                go.Scatter(x=[d(r2), d(t_end)], y=[y2, y2 + m * (t_end - r2)], mode="lines",
                           line=dict(color=LINE, width=1.5), showlegend=False, name="support"),
                row=1, col=1,
            )  # fmt: skip
    else:
        r2 = int(e["p2_row"])
        m = float(e["trigger_slope"])
        y0 = float(e["trigger_price"]) - m * (t - r2)
        fig.add_trace(
            go.Scatter(
                x=[d(r2), d(t_end)],
                y=[y0, y0 + m * (t_end - r2)],
                mode="lines",
                line=dict(color=LINE, width=1.5),
                showlegend=False,
                name="neckline",
            ),
            row=1,
            col=1,
        )
    fig.add_vline(x=d(t), line=dict(color="#888", width=1, dash="dash"))
    if show_levels:
        x0 = d(t) if after > 0 else d(max(a, t - level_days))
        for k in ("stop", "target"):
            v = float(e[k])
            if np.isfinite(v):
                _hline(fig, x0, d(t_end), v, k, LEVEL[k], dash="dash")
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=70, t=40, b=30),
        xaxis_rangeslider_visible=False,
        plot_bgcolor="white",
        bargap=0.1,
    )
    fig.update_xaxes(type="category", nticks=10, showgrid=False)
    # price axis on the right with room for its labels; volume panel labelled
    fig.update_yaxes(gridcolor="#eee", side="right", automargin=True, showticklabels=True)
    fig.update_yaxes(title_text="volume / 50d avg" if blind else "volume",
                     title_font=dict(size=10), row=2, col=1)  # fmt: skip
    fig.update_yaxes(type="log" if log_y else "linear", row=1, col=1)
    return fig


def _rel(rows: np.ndarray, t: int) -> np.ndarray:
    """Blind x labels: trading days relative to the detection day ("t-3", "t", "t+2")."""
    return np.array(["t" if r == t else f"t{r - t:+d}" for r in rows])


def _rebased(bars: pd.DataFrame, e, t: int):
    """Bars and event with prices x 100 / close at ``t``, volume / its 50-day mean up to ``t``."""
    f = 100.0 / float(bars["close"].iloc[t])
    b = bars.copy()
    b[["open", "high", "low", "close"]] = b[["open", "high", "low", "close"]] * f
    b["volume"] = b["volume"] / float(bars["volume"].iloc[max(0, t - 49) : t + 1].mean())
    e = e.copy()
    for k in PRICE_FIELDS:
        if k in e and e[k] is not None and np.isfinite(float(e[k])):
            e[k] = float(e[k]) * f
    return b, e


def _hline(fig, x0, x1, y, name, color, dash="solid", pos="top left"):
    fig.add_trace(
        go.Scatter(
            x=[x0, x1],
            y=[y, y],
            mode="lines+text",
            text=["", name],
            textposition=pos,
            line=dict(color=color, width=1.2, dash=dash),
            showlegend=False,
            name=name,
        ),
        row=1,
        col=1,
    )
