"""
Step 4b: event study + volatility test.

Reads  data/features.csv  (step 3)  and  data/prices.csv  (step 4a)
Writes data/events.csv

For every 10-K / 10-Q filing:
  car01, car03   cumulative abnormal return (stock minus XLF) over days [0,+1] and [0,+3]
                 (day 0 = first trading day on or after the filing date)
  abs_car01      size of the day-0/+1 reaction, regardless of direction
  vol_ratio      log( volatility of next 60 days / volatility of previous 60 days )

Features (all converted to within-company z-scores, because each company has its own baseline):
  drift_risk_z   risk-factor drift (10-K only)
  drift_mdna_z   MD&A drift
  tone_chg_z     change in MD&A net tone vs the previous filing of the same form
  risk_len       log length of the 10-Q risk-factor section (short = "no changes")

Then Spearman rank correlations between features and outcomes.
NOTE: ~20 tests are run, so about one p < 0.05 is expected by pure chance.

Run:  python event_study.py
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

px = pd.read_csv("data/prices.csv", index_col=0, parse_dates=True).sort_index()
ret = px.pct_change()

f = pd.read_csv("data/features.csv", dtype={"accession": str})
f = f.sort_values(["ticker", "form", "section", "filingDate"])
f["tone_chg"] = f.groupby(["ticker", "form", "section"])["net_tone"].diff()

m = f[f["section"] == "mdna"][["ticker", "form", "filingDate", "drift", "tone_chg"]]
m = m.rename(columns={"drift": "drift_mdna"})
r = f[f["section"] == "risk_factors"][["ticker", "form", "filingDate", "drift", "chars"]]
r = r.rename(columns={"drift": "drift_risk", "chars": "risk_chars"})
ev = m.merge(r, on=["ticker", "form", "filingDate"], how="inner")
ev["risk_len"] = np.log1p(ev["risk_chars"])

# within-company (and within-form) z-scores
for col, z in [("drift_mdna", "drift_mdna_z"), ("drift_risk", "drift_risk_z"),
               ("tone_chg", "tone_chg_z")]:
    g = ev.groupby(["ticker", "form"])[col]
    ev[z] = (ev[col] - g.transform("mean")) / g.transform("std")

rows = []
for _, e in ev.iterrows():
    t = e["ticker"]
    if t not in ret.columns:
        continue
    ar = (ret[t] - ret["XLF"]).dropna()  # abnormal return vs the financial-sector ETF
    pos = ar.index.searchsorted(pd.Timestamp(e["filingDate"]))
    if pos < 61 or pos + 62 > len(ar):
        continue  # not enough price history on one side
    car01 = ar.iloc[pos:pos + 2].sum()
    car03 = ar.iloc[pos:pos + 4].sum()
    back = ar.iloc[pos - 60:pos].std()
    fwd = ar.iloc[pos + 2:pos + 62].std()
    rows.append({**e.to_dict(), "car01": car01, "car03": car03,
                 "abs_car01": abs(car01), "vol_ratio": np.log(fwd / back)})

ev = pd.DataFrame(rows)
ev.to_csv("data/events.csv", index=False)
print(f"{len(ev)} filing events with price data\n")
print(ev.groupby("form").size().rename("events").to_string(), "\n")

print("Average reaction by form (CAR = abnormal return vs XLF):")
print(ev.groupby("form")[["car01", "car03", "abs_car01"]].mean().round(4), "\n")

out_rows = []
for form, g in ev.groupby("form"):
    feats = {
        "drift_mdna_z": ["abs_car01", "vol_ratio"],
        "tone_chg_z": ["car01", "car03", "vol_ratio"],
    }
    if form == "10-K":
        feats["drift_risk_z"] = ["abs_car01", "vol_ratio"]
    else:
        feats["risk_len"] = ["abs_car01", "vol_ratio"]
    for feat, outs in feats.items():
        for out in outs:
            d = g[[feat, out]].dropna()
            if len(d) < 20:
                continue
            rho, p = spearmanr(d[feat], d[out])
            out_rows.append({"form": form, "feature": feat, "outcome": out,
                             "n": len(d), "rho": round(rho, 3), "p": round(p, 3)})

res = pd.DataFrame(out_rows).sort_values("p")
print("Spearman correlations (sorted by p-value):")
print(res.to_string(index=False))
print("\nReminder: ~20 tests were run, so about one p < 0.05 is expected by chance.")
