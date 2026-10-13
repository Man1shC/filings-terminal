"""
Step 4c: robustness check for the event study.

Same tests as event_study.py, shown side by side:
  raw  = features and outcomes as they were
  qtr  = every feature and outcome demeaned within its filing quarter and form
         (e.g. all 10-Qs filed in 2023 Q2), which removes market-wide / time effects.

The sharper question the qtr columns answer:
  "In the same quarter, did the company with MORE new language than its peers
   see volatility rise MORE than its peers?"

Reads  data/events.csv  (from event_study.py)
Run:   python robustness.py
"""
import pandas as pd
from scipy.stats import spearmanr

ev = pd.read_csv("data/events.csv", parse_dates=["filingDate"])
ev["qtr"] = ev["filingDate"].dt.to_period("Q").astype(str)

cols = ["drift_mdna_z", "drift_risk_z", "tone_chg_z", "risk_len",
        "car01", "car03", "abs_car01", "vol_ratio"]
for c in cols:
    ev[c + "_q"] = ev[c] - ev.groupby(["form", "qtr"])[c].transform("mean")

print("Filing quarters and average events per quarter:")
print(ev.groupby("form")["qtr"].agg(quarters="nunique", events="size").assign(
    per_qtr=lambda d: (d["events"] / d["quarters"]).round(1)), "\n")

rows = []
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
            a = g[[feat, out]].dropna()
            b = g[[feat + "_q", out + "_q"]].dropna()
            if len(a) < 20:
                continue
            r1, p1 = spearmanr(a[feat], a[out])
            r2, p2 = spearmanr(b[feat + "_q"], b[out + "_q"])
            rows.append({
                "form": form, "feature": feat, "outcome": out, "n": len(a),
                "rho_raw": round(r1, 3), "p_raw": round(p1, 3),
                "rho_qtr": round(r2, 3), "p_qtr": round(p2, 3),
            })

res = pd.DataFrame(rows).sort_values("p_raw")
print("Raw vs quarter-demeaned Spearman correlations:")
print(res.to_string(index=False))
print("\nIf rho_qtr is close to rho_raw, the link is company-specific.")
print("If rho_qtr shrinks toward 0, it was a shared time effect.")
