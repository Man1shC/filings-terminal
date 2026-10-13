"""
Step B1: download earnings-release exhibits (8-K Item 2.02, usually Exhibit 99.1).

For each company: list EVERY 8-K with Item 2.02 since 2020-09, open the filing's
file list, grab the press-release exhibit, and save it.

Writes data/earnings/<TICKER>/<date>_<accession>.htm  and  data/earnings_index.csv
Safe to re-run (skips files already downloaded).

Run:  python fetch_earnings.py
"""
import re
from pathlib import Path

import pandas as pd

from edgar_ingest import get, ticker_to_cik_map  # reuse the polite SEC downloader

KEEP = [
    "AXP", "BAC", "C", "CFG", "COF", "EVR", "FITB", "GS", "HBAN", "HLI", "JEF",
    "JPM", "KEY", "LAZ", "MS", "MTB", "RF", "RJF", "SCHW", "STT", "TFC",
]
SINCE = "2020-09-01"  # prices start 2020-06, and we need ~60 days of history before an event
OUT = Path("data/earnings")


def earnings_filings(cik):
    """Every 8-K that reports Item 2.02 (results of operations) since SINCE."""
    sub = get(f"https://data.sec.gov/submissions/CIK{cik}.json", as_json=True)
    frames = [pd.DataFrame(sub["filings"]["recent"])]
    for f in sub["filings"].get("files", []):
        if f["filingTo"] >= SINCE:
            older = get(f"https://data.sec.gov/submissions/{f['name']}", as_json=True)
            frames.append(pd.DataFrame(older))
    df = pd.concat(frames, ignore_index=True)
    df = df[(df["form"] == "8-K") & (df["filingDate"] >= SINCE)]
    df = df[df["items"].fillna("").str.contains("2.02", regex=False)]
    return df[["filingDate", "accessionNumber", "primaryDocument"]].drop_duplicates()


def find_exhibit(cik, accession, primary):
    """Return (url, filename) of the press-release exhibit, or (None, None)."""
    base = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}/"
    listing = get(base + "index.json", as_json=True)
    files = {i["name"]: int(i.get("size") or 0) for i in listing["directory"]["item"]}
    htm = [n for n in files if n.lower().endswith((".htm", ".html"))]

    ex99 = [n for n in htm if re.search(r"(ex|exhibit)[-_ ]?99", n, re.I)]
    first = [n for n in ex99 if re.search(r"99[-_. ]?1(?!\d)", n)]  # prefer Exhibit 99.1
    if first:
        pick = first[0]
    elif ex99:
        pick = ex99[0]
    else:  # fall back to the largest HTML file that isn't the cover page
        others = [n for n in htm if n != primary and "index" not in n.lower()]
        if not others:
            return None, None
        pick = max(others, key=lambda n: files[n])
    return base + pick, pick


def main():
    cik_map = ticker_to_cik_map()
    rows = []
    for t in KEEP:
        cik = cik_map.get(t)
        if not cik:
            print(f"[skip] no CIK for {t}")
            continue
        filings = earnings_filings(cik)
        print(f"[{t}] {len(filings)} earnings 8-Ks")
        out_dir = OUT / t
        out_dir.mkdir(parents=True, exist_ok=True)

        for _, r in filings.iterrows():
            acc = r["accessionNumber"]
            path = out_dir / f"{r['filingDate']}_{acc.replace('-', '')}.htm"
            try:
                if not path.exists():
                    url, _ = find_exhibit(cik, acc, r["primaryDocument"])
                    if url is None:
                        print(f"   no exhibit found: {t} {r['filingDate']}")
                        continue
                    path.write_text(get(url), encoding="utf-8")
                rows.append({"ticker": t, "filingDate": r["filingDate"],
                             "accession": acc.replace("-", ""), "path": str(path)})
            except Exception as e:  # one bad filing shouldn't kill the run
                print(f"   [error] {t} {r['filingDate']}: {e}")

    idx = pd.DataFrame(rows)
    idx.to_csv("data/earnings_index.csv", index=False)
    print(f"\nDone. {len(idx)} earnings releases -> data/earnings_index.csv")
    print(idx.groupby("ticker").size().to_string())

    # sanity check: does a saved file look like a press release?
    jpm = idx[idx["ticker"] == "JPM"].sort_values("filingDate")
    if len(jpm):
        html = Path(jpm.iloc[-1]["path"]).read_text(encoding="utf-8", errors="ignore")
        text = " ".join(re.sub(r"<[^>]+>", " ", html).split())
        print(f"\nSample ({jpm.iloc[-1]['filingDate']} JPM):\n{text[:400]}")


if __name__ == "__main__":
    main()
