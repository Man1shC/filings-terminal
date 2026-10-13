"""
Step C2: test the LLM-extracted signals.

Reads  data/earnings_events_sue.csv (step B3) and data/llm_extractions.jsonl (step C1)
Writes data/llm_events.csv

1. Accuracy check: LLM "better/worse than last year" vs the real EPS change (sign of SUE, from XBRL)
2. Distribution of each extracted field
3. Six pre-specified Spearman tests vs the earnings reaction (raw and quarter-adjusted)
4. Regression: does the LLM add anything beyond the EPS surprise? (errors clustered by quarter)

Run:  python llm_event_study.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import spearmanr

ev = pd.read_csv("data/earnings_events_sue.csv")
rows = [json.loads(line) for line in
        Path("data/llm_extractions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
llm = pd.DataFrame(rows).drop(columns=["accession", "in_tokens", "out_tokens"])
df = ev.merge(llm, on=["ticker", "filingDate"], how="inner")
print(f"{len(df)} earnings events with both a SUE value and LLM extraction "
      f"({df['ticker'].nunique()} companies, {df['qtr'].nunique()} quarters)\n")

# ---- numeric versions of the extracted fields
df["results_num"] = df["results_vs_prior_year"].map({"better": 1, "similar": 0, "worse": -1})
df["guidance_num"] = df["guidance"].map({"raised": 1, "maintained": 0, "none": 0, "lowered": -1})
df["credit_num"] = df["credit_quality"].map(
    {"improving": 1, "stable": 0, "not_discussed": 0, "deteriorating": -1})
df["notable_num"] = df["notable_items"].map({"positive": 1, "mixed": 0, "none": 0, "negative": -1})
df["capital_num"] = df["capital_return"].astype(int)
g = df.groupby("ticker")["tone"]
df["llm_tone_z"] = (df["tone"] - g.transform("mean")) / g.transform("std")

# ---- 1. accuracy check against the real EPS change
chk = df[df["results_vs_prior_year"].isin(["better", "worse"]) & (df["sue"] != 0)].copy()
chk["eps_change"] = np.where(chk["sue"] > 0, "EPS up", "EPS down")
print("1) LLM 'results vs prior year' vs actual EPS change (from XBRL):")
print(pd.crosstab(chk["results_vs_prior_year"], chk["eps_change"]))
agree = ((chk["results_vs_prior_year"] == "better") & (chk["eps_change"] == "EPS up")) | \
        ((chk["results_vs_prior_year"] == "worse") & (chk["eps_change"] == "EPS down"))
print(f"Agreement: {agree.mean():.1%} of {len(chk)} releases "
      f"({(df['results_vs_prior_year'] == 'similar').sum()} 'similar' answers excluded)\n")

# ---- 2. distributions
print("2) How often does the LLM give each answer?")
for col in ["results_vs_prior_year", "guidance", "credit_quality", "notable_items", "capital_return"]:
    print(f"  {col}: " + ", ".join(f"{k}={v:.0%}" for k, v in
                                   df[col].value_counts(normalize=True).items()))
print(f"  tone: mean {df['tone'].mean():.2f}, "
      + ", ".join(f"{int(k)}={v:.0%}" for k, v in
                  df['tone'].value_counts(normalize=True).sort_index().items()))
rho, p = spearmanr(df["tone"], df["net_tone_z"])
print(f"  LLM tone vs lexicon tone: Spearman rho = {rho:.2f}\n")

# ---- quarter-demeaned versions
cols = ["results_num", "guidance_num", "credit_num", "notable_num", "capital_num",
        "llm_tone_z", "net_tone_z", "sue", "car01", "car03"]
for c in cols:
    df[c + "_q"] = df[c] - df.groupby("qtr")[c].transform("mean")
df.to_csv("data/llm_events.csv", index=False)

# ---- 3. six pre-specified tests
print("3) Pre-specified tests vs CAR[0,+1] (raw vs quarter-adjusted):")
res = []
for feat in ["results_num", "guidance_num", "credit_num", "notable_num", "capital_num", "llm_tone_z"]:
    a = df[[feat, "car01"]].dropna()
    b = df[[feat + "_q", "car01_q"]].dropna()
    r1, p1 = spearmanr(a[feat], a["car01"])
    r2, p2 = spearmanr(b[feat + "_q"], b["car01_q"])
    res.append({"feature": feat, "n": len(a), "rho_raw": round(r1, 3), "p_raw": round(p1, 3),
                "rho_qtr": round(r2, 3), "p_qtr": round(p2, 3)})
print(pd.DataFrame(res).to_string(index=False))
print("(6 tests: a strict cutoff is p < 0.008)\n")

df["bucket"] = pd.qcut(df["llm_tone_z_q"].rank(method="first"), 3,
                       labels=["worse tone", "middle", "better tone"])
print("Average peer-adjusted CAR[0,+1] (in %) by LLM-tone bucket:")
print((df.groupby("bucket", observed=True)["car01_q"].mean() * 100).round(3).to_string())

# ---- 4. does the LLM add anything beyond EPS surprise?
groups = pd.factorize(df["qtr"])[0]
m1 = smf.ols("car01_q ~ sue_q", data=df).fit(cov_type="cluster", cov_kwds={"groups": groups})
m2 = smf.ols("car01_q ~ sue_q + llm_tone_z_q + results_num_q + guidance_num_q + credit_num_q",
             data=df).fit(cov_type="cluster", cov_kwds={"groups": groups})
print("\n4) Regression with LLM signals added (errors clustered by quarter):")
print(m2.summary().tables[1])
print(f"\nR-squared: SUE only = {m1.rsquared:.3f}   SUE + LLM signals = {m2.rsquared:.3f}   "
      f"(adjusted: {m1.rsquared_adj:.3f} vs {m2.rsquared_adj:.3f})")
