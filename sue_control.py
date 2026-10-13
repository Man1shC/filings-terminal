"""
Step B3: positive control for the earnings event study.

SUE = standardized unexpected earnings, seasonal random-walk version:
    change   = this quarter's diluted EPS - same quarter's EPS one year earlier
    SUE      = change / std of the previous 8 such changes
A classic stand-in for "earnings surprise" when you don't have analyst consensus.

Quarterly EPS comes from the XBRL facts files saved in step 1 (data/facts/<TICKER>.json).
Q4 is derived as full-year EPS minus Q1+Q2+Q3 (10-Ks don't report Q4 on its own).

Reads  data/earnings_events.csv (step B2) and data/facts/*.json
Writes data/earnings_events_sue.csv

Prints:
  1. how many earnings events matched a SUE value
  2. positive control: SUE vs abnormal return (raw and quarter-adjusted)
  3. average peer-adjusted reaction by SUE bucket
  4. regression: does tone add anything once SUE is known? (errors clustered by quarter)

Run:  python sue_control.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

TAGS = ["EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted", "EarningsPerShareBasic"]


def quarterly_eps(path):
    """Discrete quarterly EPS series (index = quarter end date) from an XBRL facts file."""
    facts = json.loads(Path(path).read_text(encoding="utf-8"))
    us = facts.get("facts", {}).get("us-gaap", {})
    recs = None
    for tag in TAGS:
        units = us.get(tag, {}).get("units", {})
        if "USD/shares" in units:
            recs = units["USD/shares"]
            break
    if recs is None:
        return None

    d = pd.DataFrame(recs)
    d = d[d["form"].isin(["10-Q", "10-K"]) & d["start"].notna()].copy()
    if d.empty:
        return None
    d["start"] = pd.to_datetime(d["start"])
    d["end"] = pd.to_datetime(d["end"])
    d["days"] = (d["end"] - d["start"]).dt.days
    # a later filing can restate a value: keep the most recent for each period
    d = d.sort_values("filed").drop_duplicates(["start", "end"], keep="last")

    q = d[d["days"].between(80, 100)].set_index("end")["val"]  # 3-month periods only
    q = q[~q.index.duplicated(keep="last")].copy()

    # derive Q4 = full year - (Q1 + Q2 + Q3)
    for _, r in d[d["days"].between(350, 380)].iterrows():
        if r["end"] in q.index:
            continue
        inside = q[(q.index > r["start"]) & (q.index < r["end"])]
        if len(inside) == 3:
            q.loc[r["end"]] = r["val"] - inside.sum()
    return q.sort_index()


def sue_series(q):
    """Seasonal random-walk SUE, indexed by quarter end date."""
    changes = {}
    for e in q.index:
        prev = [p for p in q.index if 350 <= (e - p).days <= 380]
        if prev:
            changes[e] = q[e] - q[prev[0]]
    ch = pd.Series(changes).sort_index()
    out = {}
    vals = ch.values
    for i in range(len(ch)):
        hist = vals[max(0, i - 8):i]
        if len(hist) >= 6 and hist.std(ddof=1) > 0:
            out[ch.index[i]] = vals[i] / hist.std(ddof=1)
    return pd.Series(out, dtype=float)


def match_sue(sue, release_date):
    """SUE of the quarter that ended shortly before the release date."""
    d = pd.Timestamp(release_date)
    ok = sue[(sue.index <= d - pd.Timedelta(days=3)) & (sue.index >= d - pd.Timedelta(days=75))]
    return ok.iloc[-1] if len(ok) else np.nan


def main():
    import statsmodels.formula.api as smf
    from scipy.stats import spearmanr

    ev = pd.read_csv("data/earnings_events.csv")
    sues = {}
    for t in ev["ticker"].unique():
        p = Path(f"data/facts/{t}.json")
        q = quarterly_eps(p) if p.exists() else None
        sues[t] = sue_series(q) if q is not None and len(q) > 8 else None
        print(f"[{t}] quarters of EPS: {0 if q is None else len(q)}")

    ev["sue"] = [
        match_sue(sues[t], d) if sues[t] is not None else np.nan
        for t, d in zip(ev["ticker"], ev["filingDate"])
    ]
    ev["sue"] = ev["sue"].clip(-4, 4)
    print(f"\n{ev['sue'].notna().sum()} of {len(ev)} earnings events matched a SUE value")
    print("Matched per ticker:")
    print(ev.groupby("ticker")["sue"].apply(lambda s: f"{s.notna().sum()}/{len(s)}").to_string())

    ev = ev.dropna(subset=["sue"]).copy()
    ev["qtr"] = pd.to_datetime(ev["filingDate"]).dt.to_period("Q").astype(str)
    for c in ["sue", "net_tone_z", "car01", "car03"]:
        ev[c + "_q"] = ev[c] - ev.groupby("qtr")[c].transform("mean")
    ev.to_csv("data/earnings_events_sue.csv", index=False)

    print("\nPOSITIVE CONTROL: does SUE predict the earnings reaction?")
    rows = []
    for outc in ["car01", "car03"]:
        r1, p1 = spearmanr(ev["sue"], ev[outc])
        r2, p2 = spearmanr(ev["sue_q"], ev[outc + "_q"])
        rows.append({"feature": "sue", "outcome": outc, "n": len(ev),
                     "rho_raw": round(r1, 3), "p_raw": round(p1, 4),
                     "rho_qtr": round(r2, 3), "p_qtr": round(p2, 4)})
    print(pd.DataFrame(rows).to_string(index=False))

    ev["bucket"] = pd.qcut(ev["sue_q"].rank(method="first"), 3,
                           labels=["worse surprise", "middle", "better surprise"])
    print("\nAverage peer-adjusted CAR[0,+1] (in %) by SUE bucket:")
    print((ev.groupby("bucket", observed=True)["car01_q"].mean() * 100).round(3).to_string())

    print("\nDOES TONE ADD ANYTHING ONCE SUE IS KNOWN?  (OLS, errors clustered by quarter)")
    m = smf.ols("car01_q ~ sue_q + net_tone_z_q", data=ev).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(ev["qtr"])[0]})
    print(m.summary().tables[1])
    print(f"R-squared: {m.rsquared:.3f}   n = {int(m.nobs)}")


if __name__ == "__main__":
    main()
