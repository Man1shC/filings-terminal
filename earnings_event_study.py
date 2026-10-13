"""
Step B2: earnings-release tone vs abnormal returns, with quarter controls built in.

Reads  data/earnings_index.csv (step B1) and data/prices.csv (step 4a)
Writes data/earnings_events.csv

For every earnings release:
  tone features   neg / unc / pos rate per 1,000 words, net_tone = pos - neg
                  (same starter word lists as build_features.py)
  *_z             each feature as a z-score within that company's own history
  tone_chg        change in net_tone vs the company's previous release
  car01, car03    cumulative abnormal return (stock minus XLF) over days [0,+1] / [0,+3]
                  (day 0 = first trading day on or after the filing date; covers both
                   pre-market and after-close releases)
  *_q             every feature and outcome demeaned within its filing quarter
                  (removes market-wide / earnings-season effects)

Six pre-specified tests (Spearman), raw vs quarter-adjusted.

Run:  python earnings_event_study.py
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from scipy.stats import spearmanr

from build_features import tone_features

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)


def release_text(path):
    html = Path(path).read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")
    for t in soup(["script", "style"]):
        t.decompose()
    return " ".join(soup.get_text(" ").split())


idx = pd.read_csv("data/earnings_index.csv", dtype={"accession": str})
print(f"Reading {len(idx)} earnings releases...")

rows = []
for _, r in idx.iterrows():
    rows.append({**r.to_dict(), **tone_features(release_text(r["path"]))})
df = pd.DataFrame(rows).dropna(subset=["pos_rate"])
df["net_tone"] = df["pos_rate"] - df["neg_rate"]
df = df.sort_values(["ticker", "filingDate"]).drop_duplicates(["ticker", "filingDate"])
df["tone_chg"] = df.groupby("ticker")["net_tone"].diff()

for c in ["net_tone", "tone_chg", "neg_rate", "unc_rate"]:
    g = df.groupby("ticker")[c]
    df[c + "_z"] = (df[c] - g.transform("mean")) / g.transform("std")

px = pd.read_csv("data/prices.csv", index_col=0, parse_dates=True).sort_index()
ret = px.pct_change()

out = []
for _, e in df.iterrows():
    ar = (ret[e["ticker"]] - ret["XLF"]).dropna()  # abnormal return vs the financial-sector ETF
    pos = ar.index.searchsorted(pd.Timestamp(e["filingDate"]))
    if pos < 1 or pos + 4 > len(ar):
        continue
    car01 = ar.iloc[pos:pos + 2].sum()
    car03 = ar.iloc[pos:pos + 4].sum()
    out.append({**e.to_dict(), "car01": car01, "car03": car03, "abs_car01": abs(car01)})

ev = pd.DataFrame(out)
ev["qtr"] = pd.to_datetime(ev["filingDate"]).dt.to_period("Q").astype(str)
ev.to_csv("data/earnings_events.csv", index=False)

cols = ["net_tone_z", "tone_chg_z", "neg_rate_z", "unc_rate_z", "car01", "car03", "abs_car01"]
for c in cols:
    ev[c + "_q"] = ev[c] - ev.groupby("qtr")[c].transform("mean")

print(f"\n{len(ev)} earnings events with price data, {ev['qtr'].nunique()} quarters")
print(f"Average reaction size |CAR[0,+1]|: {ev['abs_car01'].mean():.4f} "
      f"(10-K/10-Q filing days were ~0.011 to 0.015)\n")

tests = [
    ("net_tone_z", "car01"), ("net_tone_z", "car03"),
    ("tone_chg_z", "car01"), ("tone_chg_z", "car03"),
    ("neg_rate_z", "car01"), ("unc_rate_z", "abs_car01"),
]
res = []
for feat, outc in tests:
    a = ev[[feat, outc]].dropna()
    b = ev[[feat + "_q", outc + "_q"]].dropna()
    r1, p1 = spearmanr(a[feat], a[outc])
    r2, p2 = spearmanr(b[feat + "_q"], b[outc + "_q"])
    res.append({"feature": feat, "outcome": outc, "n": len(a),
                "rho_raw": round(r1, 3), "p_raw": round(p1, 3),
                "rho_qtr": round(r2, 3), "p_qtr": round(p2, 3)})
print("Spearman correlations, raw vs quarter-adjusted:")
print(pd.DataFrame(res).to_string(index=False))

# intuition view: average peer-adjusted reaction by tone bucket
ev["bucket"] = pd.qcut(ev["net_tone_z_q"].rank(method="first"), 3,
                       labels=["worse tone", "middle", "better tone"])
print("\nAverage peer-adjusted CAR[0,+1] (in %) by tone bucket "
      "(tone vs own history, vs same-quarter peers):")
print((ev.groupby("bucket", observed=True)["car01_q"].mean() * 100).round(3).to_string())
print("\nReminder: 6 tests were run; a result only counts if rho_qtr keeps its sign and size.")
