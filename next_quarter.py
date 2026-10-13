"""
Step C3: does what management says today predict NEXT quarter's earnings surprise?

Outcome:  sue_next = the SUE of the company's next earnings release (60-120 days later)
Pre-specified tests (Spearman, raw and quarter-adjusted), 6 in total:
  sue          this quarter's surprise (baseline / positive control: earnings momentum)
  llm_tone_z   LLM tone, vs the company's own history
  credit_num   LLM credit-quality direction (improving +1 ... deteriorating -1)
  results_num  LLM results direction vs last year
  notable_num  LLM one-time-items direction
  net_tone_z   word-list tone, vs the company's own history
Strict cutoff for 6 tests: p < 0.008.

Then regressions (errors clustered by quarter):
  sue_next_q ~ sue_q                                       (baseline)
  sue_next_q ~ sue_q + llm_tone_z_q + credit_num_q + notable_num_q   (adds the LLM signals)

Reads  data/llm_events.csv (step C2)
Run:   python next_quarter.py
"""
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import spearmanr

df = pd.read_csv("data/llm_events.csv", parse_dates=["filingDate"])
df = df.sort_values(["ticker", "filingDate"])
df["next_date"] = df.groupby("ticker")["filingDate"].shift(-1)
df["sue_next"] = df.groupby("ticker")["sue"].shift(-1)
gap = (df["next_date"] - df["filingDate"]).dt.days
df = df[gap.between(60, 120)].dropna(subset=["sue_next"]).copy()  # genuine consecutive quarters only
print(f"{len(df)} release pairs (this quarter -> next quarter), "
      f"{df['ticker'].nunique()} companies, {df['qtr'].nunique()} quarters\n")

feats = ["sue", "llm_tone_z", "credit_num", "results_num", "notable_num", "net_tone_z"]
for c in feats + ["sue_next"]:  # quarter-demean on THIS sample
    df[c + "_q"] = df[c] - df.groupby("qtr")[c].transform("mean")

print("Predicting next quarter's SUE (raw vs quarter-adjusted Spearman):")
res = []
for f in feats:
    a = df[[f, "sue_next"]].dropna()
    b = df[[f + "_q", "sue_next_q"]].dropna()
    r1, p1 = spearmanr(a[f], a["sue_next"])
    r2, p2 = spearmanr(b[f + "_q"], b["sue_next_q"])
    res.append({"feature": f, "n": len(a), "rho_raw": round(r1, 3), "p_raw": round(p1, 4),
                "rho_qtr": round(r2, 3), "p_qtr": round(p2, 4)})
print(pd.DataFrame(res).to_string(index=False))
print("(6 tests: a strict cutoff is p < 0.008)\n")

df["bucket"] = pd.qcut(df["sue_q"].rank(method="first"), 3,
                       labels=["worse surprise now", "middle", "better surprise now"])
print("Average NEXT-quarter SUE (peer-adjusted) by this quarter's surprise bucket:")
print(df.groupby("bucket", observed=True)["sue_next_q"].mean().round(3).to_string())

groups = pd.factorize(df["qtr"])[0]
m1 = smf.ols("sue_next_q ~ sue_q", data=df).fit(cov_type="cluster", cov_kwds={"groups": groups})
m2 = smf.ols("sue_next_q ~ sue_q + llm_tone_z_q + credit_num_q + notable_num_q",
             data=df).fit(cov_type="cluster", cov_kwds={"groups": groups})
print("\nRegression with the LLM signals added (errors clustered by quarter):")
print(m2.summary().tables[1])
print(f"\nR-squared: baseline (SUE only) = {m1.rsquared:.3f}   with LLM signals = {m2.rsquared:.3f}   "
      f"(adjusted: {m1.rsquared_adj:.3f} vs {m2.rsquared_adj:.3f})")
