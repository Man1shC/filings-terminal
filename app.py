"""
FILINGS TERMINAL: a terminal-style research screen over SEC filings of 21 financial companies.
(Inspired by the look of professional market terminals. Not affiliated with or endorsed by any of them.)

  F1 OVERVIEW      security strip, price vs sector with earnings markers, EPS surprise, reactions,
                   risk-language drift, management tone, earnings history and LLM-extracted signals
  F2 MARKET MAP    risk-drift heatmap, surprise vs reaction scatter, earnings sensitivity ranking
  F3 RESEARCH      findings computed live from the data files, plus the evaluation of the Q&A agent
  F4 ASK           cited Q&A agent (hybrid search + filing comparison)

Run:   streamlit run app.py
Needs: data/ files from the pipeline; ANTHROPIC_API_KEY in the same window for the ASK tab only.
"""
import html
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy.stats import spearmanr

import agent as A

NAMES = {
    "AXP": "American Express", "BAC": "Bank of America", "C": "Citigroup", "CFG": "Citizens Financial",
    "COF": "Capital One", "EVR": "Evercore", "FITB": "Fifth Third", "GS": "Goldman Sachs",
    "HBAN": "Huntington Bancshares", "HLI": "Houlihan Lokey", "JEF": "Jefferies", "JPM": "JPMorgan Chase",
    "KEY": "KeyCorp", "LAZ": "Lazard", "MS": "Morgan Stanley", "MTB": "M&T Bank", "RF": "Regions Financial",
    "RJF": "Raymond James", "SCHW": "Charles Schwab", "STT": "State Street", "TFC": "Truist",
}
AMBER, WHITE, GREY, GREEN, RED, CYAN = "#FF9F1A", "#E6E6E6", "#8A8A8A", "#00D26A", "#FF3B3B", "#35C4FF"
MONO = "IBM Plex Mono, Consolas, Courier New, monospace"
MAX_QUESTIONS = 25
EXAMPLES = [
    "What new risks did JPM add in its 2026 10-K?",
    "What drove Goldman Sachs second quarter 2026 results?",
    "What did Fifth Third say about the Comerica acquisition in its April 2026 earnings release?",
    "Which banks discuss commercial real estate credit risk in their 2026 10-Ks?",
]

st.set_page_config(page_title="Filings Terminal", page_icon="▮", layout="wide", initial_sidebar_state="collapsed")
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&display=swap');
.stApp, .stApp p, .stApp label, .stApp li, .stApp td, .stApp th, .stApp h1, .stApp h2, .stApp h3,
.stApp h4, .stApp h5, .stApp button, .stApp textarea, .stApp input, .stApp [data-baseweb="select"] div
 {font-family: 'IBM Plex Mono', Consolas, 'Courier New', monospace;}
.stApp {background: #000; color: #E6E6E6;}
header[data-testid="stHeader"] {background: #000; height: 0;}
#MainMenu, footer, .stDeployButton, [data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"] {visibility: hidden; display: none;}
.block-container {padding: 0.6rem 2.2rem 3rem 2.2rem; max-width: 100%;}
.statusbar {position: fixed; left: 0; right: 0; bottom: 0; z-index: 999; background: #0A0A0A; border-top: 1px solid #333;
            padding: 3px 2.2rem; font-size: .66rem; color: #8A8A8A; letter-spacing: .06em; text-transform: uppercase;}
.statusbar b {color: #FF9F1A; font-weight: 500;} .statusbar .dot {color: #00D26A;}
.tt th.r, .tt td.r {text-align: right; font-variant-numeric: tabular-nums;}
.kv {width: 100%; border-collapse: collapse; font-size: .76rem;}
.kv td {padding: 3px 7px; border-bottom: 1px solid #141414;} .kv td:first-child {color: #8A8A8A;}
.kv td:last-child {text-align: right; color: #FFF; font-variant-numeric: tabular-nums;}
.kv tr:hover td {background: #111;}
div[data-testid="stSegmentedControl"] button {border-radius: 0 !important;}
h1, h2, h3, h4 {color: #FF9F1A; letter-spacing: .02em;}
label p {color: #FF9F1A !important; font-size: .7rem !important; letter-spacing: .1em; text-transform: uppercase;}
.brand {color: #FF9F1A; font-weight: 600; letter-spacing: .12em; font-size: .9rem; margin: 2px 0 4px 0;}
.brand span {color: #8A8A8A; font-weight: 400; margin-left: 14px;}
.tape {overflow: hidden; white-space: nowrap; border-top: 1px solid #333; border-bottom: 1px solid #333;
       background: #050505; padding: 4px 0; margin-bottom: 8px;}
.tape-inner {display: inline-block; padding-left: 100%; animation: scroll 150s linear infinite;}
.tape-inner:hover {animation-play-state: paused;}
@keyframes scroll {0% {transform: translateX(0);} 100% {transform: translateX(-100%);}}
.tape .t {margin-right: 30px; font-size: .8rem;}
.up {color: #00D26A;} .dn {color: #FF3B3B;} .am {color: #FF9F1A;} .gr {color: #8A8A8A;}
.strip {display: flex; flex-wrap: wrap; border: 1px solid #333; background: #0A0A0A; margin: 4px 0 10px 0;}
.strip .cell {padding: 6px 16px; border-right: 1px solid #222;}
.strip .lab {color: #8A8A8A; font-size: .64rem; letter-spacing: .1em; text-transform: uppercase;}
.strip .val {font-size: 1.02rem; font-weight: 600; color: #FFFFFF;}
.pt {background: #1A1A1A; color: #FF9F1A; font-weight: 600; font-size: .76rem; letter-spacing: .08em;
     padding: 3px 8px; border-left: 3px solid #FF9F1A; margin: 8px 0 2px 0; text-transform: uppercase;}
.pt span {color: #A0A0A0; font-weight: 400; margin-left: 10px; text-transform: none; letter-spacing: 0;}
.tt {width: 100%; border-collapse: collapse; font-size: .76rem;}
.tt th {color: #FF9F1A; text-align: left; border-bottom: 1px solid #333; padding: 3px 7px; font-weight: 600;
        text-transform: uppercase; font-size: .66rem; letter-spacing: .06em;}
.tt td {padding: 3px 7px; border-bottom: 1px solid #141414; color: #D6D6D6;}
.tt tr:hover td {background: #111;}
.note {color: #8A8A8A; font-size: .72rem;}
div[data-testid="stMetric"] {background: #000; border: 1px solid #2A2A2A; border-left: 3px solid #FF9F1A;
                             border-radius: 0; padding: 8px 12px;}
div[data-testid="stMetricLabel"] p {color: #8A8A8A !important;}
div[data-testid="stMetricValue"] {font-size: 1.45rem; color: #FFFFFF;}
button[data-baseweb="tab"] p {color: #8A8A8A; font-weight: 600; letter-spacing: .06em; font-size: .82rem;}
button[data-baseweb="tab"][aria-selected="true"] p {color: #FF9F1A;}
div[data-baseweb="tab-highlight"] {background-color: #FF9F1A !important;}
div[data-baseweb="tab-list"] {border-bottom: 1px solid #333;}
button[data-testid="stBaseButton-secondary"] {background: #000; color: #FF9F1A; border: 1px solid #5A3A00; border-radius: 0;}
button[data-testid="stBaseButton-secondary"]:hover {border-color: #FF9F1A; color: #FFB84D;}
button[data-testid="stBaseButton-primary"] {background: #FF9F1A; color: #000; border: none; border-radius: 0; font-weight: 600;}
div[data-testid="stExpander"] {border: 1px solid #2A2A2A; border-radius: 0; background: #050505;}
div[data-baseweb="select"] > div {background: #0A0A0A; border-radius: 0; border-color: #5A3A00;}
textarea {background: #0A0A0A !important; border-radius: 0 !important;}
</style>
""",
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------ data
@st.cache_data
def load_prices():
    p = Path("data/prices.csv")
    return pd.read_csv(p, index_col=0, parse_dates=True).sort_index() if p.exists() else None


@st.cache_data
def load_features():
    p = Path("data/features.csv")
    if not p.exists():
        return None
    df = pd.read_csv(p, dtype={"accession": str})
    df["date"] = pd.to_datetime(df["filingDate"])
    return df


@st.cache_data
def load_events():
    for name in ("data/llm_events.csv", "data/earnings_events_sue.csv"):
        if Path(name).exists():
            df = pd.read_csv(name)
            df["date"] = pd.to_datetime(df["filingDate"])
            return df.sort_values(["ticker", "date"]).reset_index(drop=True)
    return None


@st.cache_data
def load_agent_results():
    p = Path("data/eval/agent_results.jsonl")
    if not p.exists():
        return None
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    return pd.DataFrame(rows) if rows else None


@st.cache_data
def load_retrieval():
    p = Path("data/eval/retrieval_results.csv")
    return pd.read_csv(p) if p.exists() else None


# ------------------------------------------------------------------ small html helpers
def esc(x):
    return html.escape(str(x))


def sign_cls(x):
    return "" if x is None or pd.isna(x) else ("up" if x > 0 else ("dn" if x < 0 else ""))


def pct(x, d=1):
    return "n/a" if x is None or pd.isna(x) else f"{x:+.{d}%}"


def panel(code, title=""):
    st.markdown(f'<div class="pt">{esc(code)}<span>{esc(title)}</span></div>', unsafe_allow_html=True)


def cell(label, value, cls=""):
    return f'<div class="cell"><div class="lab">{esc(label)}</div><div class="val {cls}">{esc(value)}</div></div>'


def term_table(df, signed=()):
    """Dense monospace table; columns listed in `signed` are coloured green/red by their leading sign."""
    num = re.compile(r"^[+\-]?\$?[\d,]*\.?\d+(%| pts)?$")
    is_num = {c: bool(len(df)) and all((str(v) == "" or str(v) == "n/a" or num.match(str(v))) for v in df[c])
              and any(num.match(str(v)) for v in df[c]) for c in df.columns}
    head = "".join(f'<th class="{"r" if is_num[c] else ""}">{esc(c)}</th>' for c in df.columns)
    rows = []
    for _, r in df.iterrows():
        tds = []
        for c in df.columns:
            v = str(r[c])
            cls = "r " if is_num[c] else ""
            if c in signed:
                cls += "dn" if v.startswith("-") else ("up" if v.startswith("+") else "")
            tds.append(f'<td class="{cls}">{esc(v)}</td>')
        rows.append("<tr>" + "".join(tds) + "</tr>")
    st.markdown(f'<table class="tt"><tr>{head}</tr>{"".join(rows)}</table>', unsafe_allow_html=True)


def style(fig, height=340):
    fig.update_layout(
        template="plotly_dark", height=height, margin=dict(l=6, r=6, t=10, b=6),
        paper_bgcolor="#000000", plot_bgcolor="#000000", font=dict(family=MONO, color="#D0D0D0", size=11),
        legend=dict(orientation="h", y=1.08, x=0, font=dict(size=10, color="#D0D0D0")),
    )
    fig.update_xaxes(gridcolor="#1A1A1A", zeroline=False, linecolor="#333")
    fig.update_yaxes(gridcolor="#1A1A1A", zeroline=False, linecolor="#333")
    return fig


def ticker_tape(px, tickers):
    items = []
    for t in tickers + ["XLF", "SPY"]:
        if t not in px.columns:
            continue
        s = px[t].dropna()
        if len(s) < 2:
            continue
        chg = s.iloc[-1] / s.iloc[-2] - 1
        cls, arrow = ("up", "▲") if chg >= 0 else ("dn", "▼")
        items.append(f'<span class="t"><span class="am">{esc(t)}</span> {s.iloc[-1]:,.2f} '
                     f'<span class="{cls}">{arrow}{abs(chg):.2%}</span></span>')
    st.markdown(f'<div class="tape"><div class="tape-inner">{"".join(items)}</div></div>', unsafe_allow_html=True)



# ------------------------------------------------------------------ market statistics
def trailing(s, days):
    s = s.dropna()
    base = s[s.index <= s.index[-1] - pd.Timedelta(days=days)]
    return s.iloc[-1] / base.iloc[-1] - 1 if len(base) else np.nan


def ytd_ret(s):
    s = s.dropna()
    base = s[s.index < pd.Timestamp(s.index[-1].year, 1, 1)]
    return s.iloc[-1] / base.iloc[-1] - 1 if len(base) else np.nan


def returns_row(s):
    s = s.dropna()
    return [s.iloc[-1] / s.iloc[-2] - 1, trailing(s, 30), trailing(s, 91), trailing(s, 182), ytd_ret(s), trailing(s, 365)]


def perf_table(px, t):
    rows = []
    for name in (t, "XLF", "SPY"):
        if name in px.columns:
            rows.append([name] + [pct(v) for v in returns_row(px[name])])
    term_table(pd.DataFrame(rows, columns=["", "1D", "1M", "3M", "6M", "YTD", "1Y"]),
               signed=("1D", "1M", "3M", "6M", "YTD", "1Y"))


def risk_stats(px, t):
    r = px[[t, "XLF"]].dropna().pct_change().dropna()
    r = r[r.index >= r.index[-1] - pd.Timedelta(days=365)]
    s = px[t].dropna()
    y = s[s.index >= s.index[-1] - pd.Timedelta(days=365)]
    beta = r[t].cov(r["XLF"]) / r["XLF"].var()
    dd = (y / y.cummax() - 1).min()
    kv = [("Volatility (1Y, annualised)", f"{r[t].std() * np.sqrt(252):.1%}"),
          ("Beta vs XLF (1Y)", f"{beta:.2f}"), ("Correlation vs XLF", f"{r[t].corr(r['XLF']):.2f}"),
          ("Max drawdown (1Y)", f"{dd:.1%}"), ("52-week high", f"{y.max():,.2f}  ({s.iloc[-1] / y.max() - 1:+.1%})"),
          ("52-week low", f"{y.min():,.2f}  ({s.iloc[-1] / y.min() - 1:+.1%})")]
    st.markdown('<table class="kv">' + "".join(f"<tr><td>{esc(a)}</td><td>{esc(b)}</td></tr>" for a, b in kv) + "</table>",
                unsafe_allow_html=True)


def monitor(px, feats, ev, tickers):
    drift = {}
    if feats is not None:
        rf = feats[(feats["form"] == "10-K") & (feats["section"] == "risk_factors")].dropna(subset=["drift"]).sort_values("date")
        drift = rf.groupby("ticker")["drift"].last().to_dict()
    last_ev = ev.sort_values("date").groupby("ticker").last() if ev is not None else pd.DataFrame()
    rows = []
    for t in tickers:
        s = px[t].dropna()
        d1, m1, _, _, ytd, y1 = returns_row(s)
        vol = s.pct_change().tail(252).std() * np.sqrt(252)
        react = last_ev["car01"].get(t, np.nan) if len(last_ev) else np.nan
        rows.append({"_y1": y1, "Ticker": t, "Name": NAMES[t], "Last": f"{s.iloc[-1]:,.2f}", "1D": pct(d1, 2), "1M": pct(m1),
                     "YTD": pct(ytd), "1Y": pct(y1), "Vol 1Y": f"{vol:.0%}", "Last earnings": pct(react, 2),
                     "Risk drift": "n/a" if t not in drift else f"{drift[t]:.2f}"})
    df = pd.DataFrame(rows).sort_values("_y1", ascending=False).drop(columns="_y1")
    term_table(df, signed=("1D", "1M", "YTD", "1Y", "Last earnings"))


# ------------------------------------------------------------------ F1 overview
def price_chart(t, px, ev_t, start):
    s = px[t].loc[start:].dropna()
    b = px["XLF"].reindex(s.index).ffill()
    s_idx, b_idx = s / s.iloc[0] * 100, b / b.iloc[0] * 100
    rs = s_idx / b_idx * 100
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.74, 0.26], vertical_spacing=0.03)
    fig.add_trace(go.Scatter(x=s_idx.index, y=s_idx, name=t, line=dict(color=AMBER, width=2),
                             hovertemplate="%{y:.1f}<extra>" + t + "</extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=b_idx.index, y=b_idx, name="XLF (financials sector)",
                             line=dict(color="#BDBDBD", width=1.2, dash="dot"),
                             hovertemplate="%{y:.1f}<extra>XLF</extra>"), row=1, col=1)
    e = ev_t[ev_t["date"] >= s_idx.index[0]] if len(ev_t) else ev_t
    if len(e):
        pos = np.clip(s_idx.index.searchsorted(e["date"]), 0, len(s_idx) - 1)
        car = e["car01"].fillna(0).to_numpy()
        sues = e["sue"] if "sue" in e else pd.Series([np.nan] * len(e), index=e.index)
        hover = [f"EARNINGS {d:%d %b %Y}<br>Reaction vs sector {c:+.2%}"
                 + (f"<br>EPS surprise {u:+.2f}" if pd.notna(u) else "") for d, c, u in zip(e["date"], car, sues)]
        fig.add_trace(go.Scatter(
            x=s_idx.index[pos], y=s_idx.to_numpy()[pos], mode="markers", name="Earnings release",
            marker=dict(size=np.clip(8 + 140 * np.abs(car), 8, 22), color=[GREEN if c >= 0 else RED for c in car],
                        line=dict(width=1, color="#000"), opacity=0.95),
            text=hover, hoverinfo="text"), row=1, col=1)
    fig.add_trace(go.Scatter(x=rs.index, y=rs, name="Relative strength vs XLF", line=dict(color=CYAN, width=1.6),
                             hovertemplate="%{y:.1f}<extra>Rel. strength</extra>"), row=2, col=1)
    fig.add_hline(y=100, line=dict(color="#555", width=1, dash="dot"), row=2, col=1)
    fig.update_yaxes(side="right", row=1, col=1)
    fig.update_yaxes(side="right", title_text="Rel. strength", title_font=dict(size=10), rangemode="normal", autorange=True, row=2, col=1)
    fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor", spikecolor="#707070",
                     spikethickness=1, spikedash="dot")
    fig.update_layout(hovermode="x unified", hoverlabel=dict(bgcolor="#111", font=dict(family=MONO, size=11)))
    return style(fig, 470)


def overview(t, px, feats, ev, start):
    ev_t = ev[ev["ticker"] == t] if ev is not None else pd.DataFrame()
    f_t = feats[feats["ticker"] == t] if feats is not None else pd.DataFrame()
    s, x = px[t].dropna(), px["XLF"].dropna()
    year = s[s.index >= s.index[-1] - pd.Timedelta(days=365)]
    xyear = x[x.index >= x.index[-1] - pd.Timedelta(days=365)]
    d1, r1, rx = s.iloc[-1] / s.iloc[-2] - 1, s.iloc[-1] / year.iloc[0] - 1, x.iloc[-1] / xyear.iloc[0] - 1

    cells = [cell("Security", f"{t} US Equity", "am"), cell("Name", NAMES.get(t, t)),
             cell("Last", f"{s.iloc[-1]:,.2f}"), cell("1D", pct(d1, 2), sign_cls(d1)),
             cell("1Y", pct(r1), sign_cls(r1)), cell("vs XLF 1Y", f"{(r1 - rx) * 100:+.1f} pts", sign_cls(r1 - rx))]
    if len(ev_t):
        last = ev_t.iloc[-1]
        cells.append(cell(f"Last earnings {last['date']:%d %b %y}", pct(last["car01"], 2), sign_cls(last["car01"])))
        if "sue" in last and pd.notna(last["sue"]):
            cells.append(cell("EPS surprise", f"{last['sue']:+.1f}", sign_cls(last["sue"])))
    rf = pd.DataFrame()
    if feats is not None and len(f_t):
        rf = f_t[(f_t["form"] == "10-K") & (f_t["section"] == "risk_factors")].dropna(subset=["drift"]).sort_values("date")
    if len(rf):
        peers = feats[(feats["form"] == "10-K") & (feats["section"] == "risk_factors")
                      & (feats["date"].dt.year == rf.iloc[-1]["date"].year)]["drift"].median()
        cells.append(cell("Risk drift (10-K)", f"{rf.iloc[-1]['drift']:.2f}"))
        cells.append(cell("vs peer median", f"{rf.iloc[-1]['drift'] - peers:+.2f}"))
    if len(ev_t) and "tone" in ev_t and ev_t["tone"].notna().any():
        cells.append(cell("Tone (LLM)", f"{ev_t.iloc[-1]['tone']:.0f} / 5"))
    st.markdown(f'<div class="strip">{"".join(cells)}</div>', unsafe_allow_html=True)

    left, right = st.columns([5, 3], gap="small")
    with left:
        panel("GP", f"{t} price vs sector, indexed to 100  ·  circles = earnings releases (green/red = reaction vs sector)  ·  lower panel = relative strength")
        st.plotly_chart(price_chart(t, px, ev_t, start), width="stretch")
    with right:
        panel("PERF", "total price return vs sector and market")
        perf_table(px, t)
        panel("RISK", "risk statistics")
        risk_stats(px, t)
        panel("ERN", "earnings history")
        if len(ev_t):
            h = ev_t.sort_values("date", ascending=False).head(10)
            tbl = pd.DataFrame({
                "Date": h["date"].dt.strftime("%d %b %y"),
                "SUE": h["sue"].map(lambda v: "n/a" if pd.isna(v) else f"{v:+.1f}") if "sue" in h else "n/a",
                "React": (h["car01"] * 100).map(lambda v: f"{v:+.2f}%"),
                "vs PY": h["results_vs_prior_year"].fillna("") if "results_vs_prior_year" in h else "",
                "Tone": h["tone"].map(lambda v: "" if pd.isna(v) else f"{v:.0f}") if "tone" in h else "",
            })
            term_table(tbl, signed=("SUE", "React"))
        else:
            st.markdown('<p class="note">No earnings data found (data/llm_events.csv).</p>', unsafe_allow_html=True)

    if len(ev_t):
        e = ev_t[ev_t["date"] >= start]
        c1, c2, c3 = st.columns(3, gap="small")
        with c1:
            panel("SUE", "eps surprise vs same quarter last year, standardised")
            if "sue" in e and e["sue"].notna().any():
                fig = go.Figure(go.Bar(x=e["date"], y=e["sue"], marker_color=[GREEN if v >= 0 else RED for v in e["sue"].fillna(0)],
                                       hovertemplate="%{x|%b %Y}<br>SUE %{y:.2f}<extra></extra>"))
                st.plotly_chart(style(fig, 250), width="stretch")
        with c2:
            panel("REACT", "stock move on earnings, day 0 to +1, vs sector")
            fig = go.Figure(go.Bar(x=e["date"], y=e["car01"] * 100, marker_color=[GREEN if v >= 0 else RED for v in e["car01"].fillna(0)],
                                   hovertemplate="%{x|%b %Y}<br>%{y:.2f}%<extra></extra>"))
            fig.update_yaxes(ticksuffix="%")
            st.plotly_chart(style(fig, 250), width="stretch")
        with c3:
            panel("TONE", "management tone in releases, llm-rated 1 to 5")
            et = e.dropna(subset=["tone"]) if "tone" in e else pd.DataFrame()
            if len(et):
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=et["date"], y=et["tone"], mode="markers+lines", name="Tone",
                                         line=dict(color="#555", width=1), marker=dict(color=AMBER, size=6)))
                fig.add_trace(go.Scatter(x=et["date"], y=et["tone"].rolling(4, min_periods=2).mean(), name="4Q avg",
                                         line=dict(color=CYAN, width=2)))
                fig.update_yaxes(range=[0.8, 5.2])
                st.plotly_chart(style(fig, 250), width="stretch")

    c1, c2 = st.columns([2, 3], gap="small")
    with c1:
        panel("RSK", "risk-language drift by 10-K: share of new wording vs prior year")
        if len(rf):
            peer = (feats[(feats["form"] == "10-K") & (feats["section"] == "risk_factors")]
                    .assign(y=lambda d: d["date"].dt.year).groupby("y")["drift"].median())
            fig = go.Figure()
            fig.add_trace(go.Bar(x=rf["date"].dt.year.astype(str), y=rf["drift"], name=t, marker_color=AMBER,
                                 hovertemplate="%{x}<br>drift %{y:.2f}<extra></extra>"))
            fig.add_trace(go.Scatter(x=peer.index.astype(str), y=peer.values, name="Peer median", mode="lines+markers",
                                     line=dict(color=CYAN, dash="dot")))
            st.plotly_chart(style(fig, 290), width="stretch")
        else:
            st.markdown('<p class="note">No risk-drift data found (data/features.csv).</p>', unsafe_allow_html=True)
    with c2:
        panel("SIG", "latest earnings releases: signals extracted by the llm")
        if len(ev_t) and "driver" in ev_t:
            h = ev_t.sort_values("date", ascending=False).head(7)
            tbl = pd.DataFrame({
                "Date": h["date"].dt.strftime("%d %b %y"),
                "Guidance": h["guidance"] if "guidance" in h else "",
                "Credit": h["credit_quality"] if "credit_quality" in h else "",
                "Main driver": h["driver"].astype(str).str.slice(0, 95),
            })
            term_table(tbl)


# ------------------------------------------------------------------ F2 market map
def market_map(px, feats, ev, tickers):
    panel("MON", "monitor: all 21 companies, sorted by 1-year return")
    monitor(px, feats, ev, tickers)
    if feats is not None:
        rf = feats[(feats["form"] == "10-K") & (feats["section"] == "risk_factors")].dropna(subset=["drift"])
        piv = rf.assign(y=rf["date"].dt.year).pivot_table(index="ticker", columns="y", values="drift")
        if not piv.empty:
            panel("HEAT", "risk-language drift by company and 10-K filing year  ·  brighter = more of the Risk Factors text was rewritten")
            fig = go.Figure(go.Heatmap(
                z=piv.values, x=[str(c) for c in piv.columns], y=list(piv.index),
                colorscale=[[0, "#000000"], [0.5, "#7A4A00"], [1, "#FFB000"]], colorbar=dict(title="drift", thickness=10),
                text=np.round(piv.values, 2), texttemplate="%{text}", textfont=dict(size=10, color="#FFFFFF"),
                hovertemplate="%{y} · %{x}<br>drift %{z:.2f}<extra></extra>"))
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(style(fig, 560), width="stretch")
            st.markdown('<p class="note">JPM\'s 2026 jump is a genuine rewrite (confirmed with a sentence-level comparison), not a parsing artefact.</p>',
                        unsafe_allow_html=True)
    if ev is None:
        return
    left, right = st.columns([3, 2], gap="small")
    with left:
        panel("XY", "does a bigger eps surprise mean a bigger stock move?")
        adj = st.toggle("Remove season and market effects (compare each company with peers in the same quarter)", value=True)
        xcol, ycol = ("sue_q", "car01_q") if adj and "sue_q" in ev else ("sue", "car01")
        d = ev.dropna(subset=[xcol, ycol])
        fig = go.Figure(go.Scatter(x=d[xcol], y=d[ycol] * 100, mode="markers", text=d["ticker"] + " " + d["date"].dt.strftime("%b %Y"),
                                   marker=dict(color=AMBER, size=6, opacity=0.55),
                                   hovertemplate="%{text}<br>surprise %{x:.2f}<br>reaction %{y:.2f}%<extra></extra>"))
        if len(d) > 5:
            m, b0 = np.polyfit(d[xcol], d[ycol] * 100, 1)
            xs = np.linspace(d[xcol].min(), d[xcol].max(), 20)
            fig.add_trace(go.Scatter(x=xs, y=m * xs + b0, mode="lines", line=dict(color=CYAN, width=2.2)))
            rho, p = spearmanr(d[xcol], d[ycol])
            st.markdown(f'<p class="note">Spearman rank correlation {rho:+.2f}  (p = {p:.3f}, n = {len(d)}). '
                        "Earnings surprise moves stocks, but only modestly.</p>", unsafe_allow_html=True)
        fig.update_xaxes(title="EPS surprise (SUE)")
        fig.update_yaxes(title="Reaction, day 0 to +1", ticksuffix="%")
        fig.update_layout(showlegend=False)
        st.plotly_chart(style(fig, 420), width="stretch")
    with right:
        panel("RANK", "earnings sensitivity: average size of the stock move")
        sens = ev.groupby("ticker")["car01"].apply(lambda s: s.abs().mean() * 100).sort_values()
        fig = go.Figure(go.Bar(x=sens.values, y=sens.index, orientation="h", marker_color=AMBER,
                               hovertemplate="%{y}: %{x:.2f}%<extra></extra>"))
        fig.update_xaxes(ticksuffix="%")
        st.plotly_chart(style(fig, 500), width="stretch")


# ------------------------------------------------------------------ F3 research
def research(ev):
    panel("RES", "what the research found  ·  every number is computed from the pipeline's output files; not investment advice")
    if ev is not None and "sue_q" in ev:
        d = ev.dropna(subset=["sue_q", "car01_q"])
        rho, p = spearmanr(d["sue_q"], d["car01_q"])
        terc = pd.qcut(d["sue_q"].rank(method="first"), 3, labels=["worse", "middle", "better"])
        spread = d.groupby(terc, observed=True)["car01_q"].mean()
        c1, c2, c3, c4 = st.columns(4, gap="small")
        c1.metric("1. Surprise → reaction (season-adjusted)", f"ρ = {rho:+.2f}", f"p = {p:.3f}, n = {len(d)}", delta_color="off")
        c2.metric("Best vs worst surprise tercile", f"{(spread['better'] - spread['worse']) * 100:+.2f} pts",
                  "avg peer-adjusted reaction", delta_color="off")
        for col, label, c in ((c3, "2. LLM tone → reaction", "llm_tone_z_q"), (c4, "Word-list tone → reaction", "net_tone_z_q")):
            if c in ev:
                dd = ev.dropna(subset=[c, "car01_q"])
                r, pp = spearmanr(dd[c], dd["car01_q"])
                col.metric(label, f"ρ = {r:+.2f}", f"p = {pp:.2f}, no detectable effect" if pp > 0.05 else f"p = {pp:.3f}", delta_color="off")
        if "results_vs_prior_year" in ev:
            chk = ev[ev["results_vs_prior_year"].isin(["better", "worse"]) & (ev["sue"] != 0)].dropna(subset=["sue"])
            agree = (((chk["results_vs_prior_year"] == "better") & (chk["sue"] > 0))
                     | ((chk["results_vs_prior_year"] == "worse") & (chk["sue"] < 0))).mean()
            st.metric("3. LLM extraction accuracy vs actual EPS direction (XBRL)", f"{agree:.1%}", f"{len(chk)} earnings releases", delta_color="off")
    st.markdown(
        "**Reading these together:** EPS surprise moves stocks, modestly. Neither an LLM nor a word-list reading of the "
        "language adds anything on top of the numbers. An apparent volatility signal that looked strongly significant "
        "(p = 0.001) disappeared once each filing was compared only with peers from the same quarter."
    )

    panel("EVAL", "the q&a agent, evaluated")
    ar, rr = load_agent_results(), load_retrieval()
    left, right = st.columns(2, gap="small")
    with left:
        if ar is not None:
            order = ["correct", "partial", "incorrect", "no_answer"]
            g = pd.DataFrame({"With filings (agent)": ar["agent_grade"].value_counts(normalize=True),
                              "Same model, no documents": ar["baseline_grade"].value_counts(normalize=True)}).reindex(order).fillna(0) * 100
            fig = go.Figure()
            for col, color in (("With filings (agent)", AMBER), ("Same model, no documents", "#6B6B6B")):
                fig.add_trace(go.Bar(x=["Correct", "Partial", "Incorrect", "No answer"], y=g[col], name=col, marker_color=color))
            fig.update_layout(barmode="group")
            fig.update_yaxes(ticksuffix="%")
            st.plotly_chart(style(fig, 320), width="stretch")
            claims = ar["n_claims"].sum()
            st.markdown(f'<p class="note">{len(ar)} questions graded against reference answers. Faithfulness: '
                        f"{ar['n_supported'].sum() / claims:.1%} of {int(claims)} claims were stated by the passages cited next to them; "
                        f"{int((ar['invalid_citations'].map(len) > 0).sum())} answers cited a source that was never retrieved. "
                        "Small sample: treat percentages as approximate.</p>", unsafe_allow_html=True)
        else:
            st.info("Run eval_agent.py to see the answer evaluation here.")
    with right:
        if rr is not None:
            rr = rr.copy()
            for c in ("hit@1", "hit@5", "hit@10", "doc_hit@5"):
                rr[c] = (rr[c] * 100).round(1).astype(str) + "%"
            rr["MRR"] = rr["MRR"].round(3).astype(str)
            st.markdown('<p class="note">Retrieval quality: does search find the right passage? hit@k = in the top k results.</p>',
                        unsafe_allow_html=True)
            term_table(rr.rename(columns={"setting": "Setting", "mode": "Search mode"}))
            st.markdown('<p class="note">Setting company, form and filing month as filters lifts hit@5 from roughly 40-60% to 87-96%.</p>',
                        unsafe_allow_html=True)

    panel("LIM", "limitations")
    st.markdown(
        "- Evaluation questions were written by an LLM from the passages themselves, which favours keyword matching and is easier than real user questions.\n"
        "- Small samples: 502 earnings events and a 30-question answer evaluation. Differences of a few points are within noise.\n"
        "- EPS surprise is the seasonal random-walk version, not analyst-consensus surprise.\n"
        "- Filings from 2025 onward are searchable; flattened HTML tables can attach a number to the wrong period."
    )


# ------------------------------------------------------------------ F4 ask
def style_citations(text):
    return re.sub(r"\[([CD]\d+(?:\.\d+)?(?:\s*[,;]\s*[CD]\d+(?:\.\d+)?)*)\]", r"`[\1]`", text)


def ask_tab():
    panel("ASK", "hybrid search plus a filing-comparison tool  ·  every claim is cited to its source passage")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        st.warning('No API key found. Close the app, run  $env:ANTHROPIC_API_KEY = "your-key"  in the same PowerShell window, '
                   "and start it again. The other tabs work without a key.")
        return
    if not Path("data/index/emb.npy").exists():
        st.warning("Search index not found. Run build_index.py first.")
        return
    st.session_state.setdefault("question", "")
    st.session_state.setdefault("count", 0)

    def use_example(text):
        st.session_state.question = text

    cols = st.columns(len(EXAMPLES), gap="small")
    for col, ex in zip(cols, EXAMPLES):
        col.button(ex, key=f"ex_{ex}", on_click=use_example, args=(ex,), width="stretch")
    q = st.text_area("Question", key="question", height=80,
                     placeholder="e.g. What new risks did Morgan Stanley add in its 2026 10-K?")
    model = st.selectbox("Model", [A.MODEL, "claude-haiku-5-5"], help="Haiku is cheaper and faster.")
    if st.button("RUN", type="primary"):
        if not q.strip():
            st.warning("Type a question first.")
        elif st.session_state.count >= MAX_QUESTIONS:
            st.warning(f"Session limit of {MAX_QUESTIONS} questions reached. Restart the app to continue.")
        else:
            with st.spinner("Searching the filings (the first question also loads the search index)..."):
                t0 = time.time()
                try:
                    st.session_state.last = {"r": A.ask(q.strip(), model), "secs": time.time() - t0, "model": model}
                    st.session_state.count += 1
                except Exception as e:
                    st.session_state.pop("last", None)
                    st.error(f"Something went wrong: {type(e).__name__}: {e}")
    st.markdown(f'<p class="note">Questions this session: {st.session_state.count} / {MAX_QUESTIONS}. '
                "Each costs roughly a few cents on your API key.</p>", unsafe_allow_html=True)

    last = st.session_state.get("last")
    if last:
        r = last["r"]
        left, right = st.columns([3, 2], gap="small")
        with left:
            panel("ANS", "answer")
            st.markdown(style_citations(r["answer"]))
            if r["invalid_citations"]:
                st.warning("These citations point to sources that were never retrieved: " + ", ".join(r["invalid_citations"]))
        with right:
            panel("TRC", "how the agent worked")
            for i, c in enumerate(r["tool_calls"], 1):
                args = ", ".join(f"{k}={json.dumps(v)}" for k, v in c["args"].items())
                st.markdown(f"{i}. **{c['tool']}**({args})")
            if not r["tool_calls"]:
                st.write("It answered without using any tools.")
            st.markdown(f'<p class="note">{len(r["tool_calls"])} tool calls · {len(r["cited"])} sources cited · '
                        f'{last["secs"]:.0f}s · {esc(last["model"])}</p>', unsafe_allow_html=True)
        panel("SRC", "sources cited")
        shown = 0
        for c in r["cited"]:
            src = r["sources"].get(c)
            if src:
                shown += 1
                with st.expander(f"[{c}]  {src['label']}"):
                    st.write(src.get("text") or "(the full comparison is in the tool result)")
        if not shown:
            st.write("The answer cites no sources.")


# ------------------------------------------------------------------ page
px, feats, ev = load_prices(), load_features(), load_events()
st.markdown('<div class="brand">FILINGS TERMINAL<span>SEC FILINGS ANALYTICS  ·  21 FINANCIAL COMPANIES  ·  RESEARCH TOOL, NOT INVESTMENT ADVICE</span></div>',
            unsafe_allow_html=True)
if px is None:
    st.error("data/prices.csv not found. Run fetch_prices.py first.")
    st.stop()
tickers = [t for t in NAMES if t in px.columns]
ticker_tape(px, tickers)

c1, c2, c3 = st.columns([3, 3, 4], gap="small")
with c1:
    t = st.selectbox("Command ▸ security", tickers, index=tickers.index("JPM") if "JPM" in tickers else 0,
                     format_func=lambda x: f"{x} US Equity  ·  {NAMES[x]}")
with c2:
    span = st.segmented_control("Range", ["6M", "1Y", "2Y", "3Y", "Max"], default="2Y") or "2Y"
days = {"6M": 183, "1Y": 365, "2Y": 730, "3Y": 1095}.get(span)
start = px.index[-1] - pd.Timedelta(days=days) if days else px.index[0]

n_f = 0 if feats is None else feats["accession"].nunique()
n_e = 0 if ev is None else len(ev)
st.markdown(f'<div class="statusbar"><span class="dot">●</span> <b>PRICES</b> THROUGH {px.index[-1]:%d %b %Y}  ·  '
            f'<b>FILINGS</b> {n_f:,} 10-K/10-Q  ·  <b>EARNINGS EVENTS</b> {n_e}  ·  <b>SOURCES</b> SEC EDGAR, YAHOO FINANCE  ·  '
            'RESEARCH TOOL, NOT INVESTMENT ADVICE</div>', unsafe_allow_html=True)
tab1, tab2, tab3, tab4 = st.tabs(["F1  OVERVIEW", "F2  MARKET MAP", "F3  RESEARCH", "F4  ASK THE FILINGS"])
with tab1:
    overview(t, px, feats, ev, start)
with tab2:
    market_map(px, feats, ev, tickers)
with tab3:
    research(ev)
with tab4:
    ask_tab()
