"""
Step 1: EDGAR ingest.

For each ticker:
  1. ticker -> CIK
  2. pull the submissions index (all recent filings)
  3. keep 10-K / 10-Q / 8-K since a start date
  4. download the main document of each filing
  5. pull XBRL company facts (structured financials for trend features)

Outputs:
  data/raw/<TICKER>/<form>_<filingDate>_<accession>.htm
  data/facts/<TICKER>.json
  data/filings_index.csv   (one row per filing, used by every later step)

Run:  python edgar_ingest.py
"""
import json
import time
from pathlib import Path

import pandas as pd
import requests

# SEC requires a real contact in the User-Agent. Change this.
HEADERS = {"User-Agent": "Manish Cheeti manishcheeti@gmail.com"}

TICKERS = ["JPM", "GS", "MS", "BAC", "C", "WFC", "USB", "PNC", "TFC", "SCHW", "BK", "BNY", "STT", "BLK", "AXP", "COF", "NTRS", "FITB", "RF", "KEY", "HBAN", "CFG", "MTB", "EVR", "PJT", "HLI", "MC", "LAZ", "JEF", "SF", "RJF"]
FORMS = {"10-K", "10-Q", "8-K"}
SINCE = "2019-01-01"
MAX_PER_FORM = {"10-K": 6, "10-Q": 20, "8-K": 60}  # keeps first run fast

DATA = Path("data")
RAW = DATA / "raw"
FACTS = DATA / "facts"


def get(url, as_json=False, retries=6):
    """GET with polite rate limiting + retries (SEC allows ~10 req/s)."""
    for attempt in range(retries):
        time.sleep(0.15)
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            time.sleep(5 * (attempt + 1))  # connection dropped: wait, then retry
            continue
        if r.status_code == 200:
            return r.json() if as_json else r.text
        if r.status_code in (429, 503):
            time.sleep(2 * (attempt + 1))
            continue
        r.raise_for_status()
    raise RuntimeError(f"Failed after {retries} tries: {url}")


def ticker_to_cik_map():
    data = get("https://www.sec.gov/files/company_tickers.json", as_json=True)
    return {v["ticker"].upper(): str(v["cik_str"]).zfill(10) for v in data.values()}


def list_filings(cik):
    """Turn the submissions JSON into a tidy DataFrame of recent filings."""
    sub = get(f"https://data.sec.gov/submissions/CIK{cik}.json", as_json=True)
    frames = [pd.DataFrame(sub["filings"]["recent"])]
    for f in sub["filings"].get("files", []):
        if f["filingTo"] >= SINCE:
            older = get(f"https://data.sec.gov/submissions/{f['name']}", as_json=True)
            frames.append(pd.DataFrame(older))
    df = pd.concat(frames, ignore_index=True)
    df = df[df["form"].isin(FORMS)]
    df = df[df["filingDate"] >= SINCE]
    df = df.sort_values("filingDate", ascending=False)
    # cap per form so the first run isn't huge
    parts = [df[df["form"] == f].head(n) for f, n in MAX_PER_FORM.items()]
    keep = pd.concat(parts)
    keep["cik"] = cik
    return keep[
        ["cik", "form", "filingDate", "reportDate", "accessionNumber",
         "primaryDocument", "items"]
    ]


def download_filing(ticker, row):
    """Download one filing's main document. Skips if already on disk."""
    acc_nodash = row["accessionNumber"].replace("-", "")
    fname = f"{row['form']}_{row['filingDate']}_{acc_nodash}.htm"
    out = RAW / ticker / fname
    if out.exists():
        return str(out)
    url = (
        f"https://www.sec.gov/Archives/edgar/data/{int(row['cik'])}/"
        f"{acc_nodash}/{row['primaryDocument']}"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(get(url), encoding="utf-8")
    return str(out)


def download_facts(ticker, cik):
    """XBRL company facts: structured quarterly/annual numbers."""
    out = FACTS / f"{ticker}.json"
    if out.exists():
        return
    FACTS.mkdir(parents=True, exist_ok=True)
    facts = get(
        f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json", as_json=True
    )
    out.write_text(json.dumps(facts), encoding="utf-8")


def main():
    cik_map = ticker_to_cik_map()
    all_rows = []

    for t in TICKERS:
        cik = cik_map.get(t)
        if not cik:
            print(f"[skip] no CIK for {t}")
            continue

        try:
            print(f"[{t}] listing filings...")
            filings = list_filings(cik)
            filings["ticker"] = t

            paths = []
            for _, row in filings.iterrows():
                paths.append(download_filing(t, row))
            filings["path"] = paths

            download_facts(t, cik)
            all_rows.append(filings)
            print(f"[{t}] {len(filings)} filings saved")
        except Exception as e:  # one bad ticker shouldn't kill the whole run
            print(f"[error] {t}: {e} (re-run the script to resume)")

    index = pd.concat(all_rows, ignore_index=True)
    DATA.mkdir(exist_ok=True)
    index.to_csv(DATA / "filings_index.csv", index=False)
    print(f"\nDone. {len(index)} filings indexed -> data/filings_index.csv")
    print(index.groupby(["ticker", "form"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
