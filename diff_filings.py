"""
Sentence-level diff of one section between two filings.

Run:  python diff_filings.py JPM 10-K risk_factors 2025 2026
      (last two args are filing-date prefixes, e.g. 2025 or 2025-02)

Shows how many sentences were kept / removed / added, plus a spread of examples.
Reworded sentences appear once as removed and once as added.
"""
import glob
import re
import sys
from pathlib import Path

ticker, form, section, p1, p2 = sys.argv[1:6]

NAMES = re.compile(
    r"jpmorgan\s*chase|jpmorgan|bank of america|goldman sachs|morgan stanley"
    r"|citigroup|citibank|\bciti\b",
    re.I,
)


def load(prefix):
    files = sorted(glob.glob(f"data/sections/{ticker.upper()}/{form}_{prefix}*_{section}.txt"))
    if not files:
        sys.exit(f"No file for {ticker} {form} {section} {prefix}")
    return Path(files[0]).read_text(encoding="utf-8"), Path(files[0]).name


def sentences(text):
    parts = re.split(r"(?<=[.!?])\s+|\n", text)
    out = []
    for s in parts:
        s = " ".join(s.split())
        if len(s) >= 60:
            norm = re.sub(r"\d+", "#", NAMES.sub("", s.lower()))
            out.append((norm, s))
    return out


a_text, a_name = load(p1)
b_text, b_name = load(p2)
A, B = sentences(a_text), sentences(b_text)
set_a, set_b = {n for n, _ in A}, {n for n, _ in B}

removed = [s for n, s in A if n not in set_b]
added = [s for n, s in B if n not in set_a]
kept = len([1 for n, _ in A if n in set_b])

print(f"{a_name}\n{b_name}")
print(f"\nsentences: {len(A)} -> {len(B)}")
print(f"kept: {kept}  removed: {len(removed)}  added: {len(added)}\n")


def show(title, items, n=8):
    print(title)
    step = max(1, len(items) // n)
    for s in items[::step][:n]:
        print("  -", s[:170])
    print()


show("REMOVED (sampled across the section):", removed)
show("ADDED (sampled across the section):", added)
