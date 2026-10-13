"""
Show what follows each "Item 1A" heading in the latest 10-Q of the given tickers.

Run:  python peek_10q_risk.py BAC GS MS C
"""
import re
import sys

import pandas as pd

import parse_sections as p

df = pd.read_csv("data/filings_index.csv")
for t in sys.argv[1:]:
    t = t.upper()
    r = df[(df.ticker == t) & (df.form == "10-Q")].sort_values("filingDate").iloc[-1]
    text = p.html_to_text(r["path"])
    print(f"\n=== {t} 10-Q {r['filingDate']} ===")
    hits = list(re.finditer(r"^\s*(item\s*1a|risk\s*factors)", text, re.I | re.M))
    if hits:
        shows = [(m.start(), text[m.start():m.start() + 400]) for m in hits[-5:]]
    else:
        print("(no line-start heading; showing last 'risk factors' mentions)")
        loose = list(re.finditer(r"risk\s*factors", text, re.I))
        shows = [
            (m.start(), text[max(0, m.start() - 150):m.start() + 250])
            for m in loose[-4:]
        ]
    for pos, chunk in shows:
        print(f"@{pos}: {' '.join(chunk.split())}\n")
