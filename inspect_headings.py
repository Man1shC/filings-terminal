"""
Print heading-like lines from the latest 10-K (or 10-Q) of the given tickers,
so we can see how each bank labels Risk Factors / MD&A.

Run:  python inspect_headings.py BAC JPM MS          (latest 10-K)
      python inspect_headings.py 10-Q BAC GS MS      (latest 10-Q)
"""
import re
import sys

import pandas as pd

import parse_sections as p

args = sys.argv[1:]
form = "10-K"
if args and args[0] in ("10-K", "10-Q"):
    form, args = args[0], args[1:]

items = "1a|1b|7|7a|8" if form == "10-K" else "1a|2|3|4"
pat = re.compile(
    r"management\W?s\s+discussion|risk\s*factors|quantitative\s+and\s+qualitative"
    r"|unresolved\s+staff|financial\s+statements\s+and\s+supplementary"
    r"|^\s*properties\s*$|^\s*item\s*(" + items + r")\b",
    re.I,
)

df = pd.read_csv("data/filings_index.csv")
for t in args:
    t = t.upper()
    r = df[(df.ticker == t) & (df.form == form)].sort_values("filingDate").iloc[-1]
    lines = p.html_to_text(r["path"]).split("\n")
    print(f"\n=== {t} {form} {r['filingDate']}  ({len(lines)} lines) ===")
    shown = 0
    for i, line in enumerate(lines):
        s = line.strip()
        if len(s) < 70 and pat.search(s):
            print(i, repr(s[:70]))
            shown += 1
            if shown >= 50:
                break
