"""
Step 4a: download daily prices for the event study.

Writes data/prices.csv  (one column per ticker, adjusted close, indexed by date)

Benchmarks:
  XLF = financial-sector ETF (what a bank "should" do on a normal day)
  SPY = whole market

Run:  python fetch_prices.py
"""
from pathlib import Path

import pandas as pd
import yfinance as yf

KEEP = [
    "AXP", "BAC", "C", "CFG", "COF", "EVR", "FITB", "GS", "HBAN", "HLI", "JEF",
    "JPM", "KEY", "LAZ", "MS", "MTB", "RF", "RJF", "SCHW", "STT", "TFC",
]
TICKERS = KEEP + ["XLF", "SPY"]

px = yf.download(TICKERS, start="2020-06-01", auto_adjust=True, progress=False)["Close"]
px = px.sort_index()

Path("data").mkdir(exist_ok=True)
px.to_csv("data/prices.csv")

print(f"Saved data/prices.csv: {px.shape[0]} trading days x {px.shape[1]} tickers")
print(f"Date range: {px.index.min().date()} -> {px.index.max().date()}")

missing = px.isna().sum()
missing = missing[missing > 0]
if len(missing):
    print("\nTickers with missing days (some start later, that's fine):")
    print(missing.sort_values(ascending=False))
else:
    print("\nNo missing days.")

print("\nLast 3 rows:")
print(px[["JPM", "GS", "XLF", "SPY"]].tail(3).round(2))
