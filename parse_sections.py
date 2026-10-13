"""
Step 2: parse filings into clean text + sections.

Reads  data/filings_index.csv  (from step 1)
Writes data/sections/<TICKER>/<form>_<date>_<acc>_<section>.txt
       data/sections_index.csv   (one row per extracted section, with char counts)

10-K / 10-Q -> risk_factors, mdna
8-K         -> full text (we split by item later)

Run:  python parse_sections.py            (all tickers)
      python parse_sections.py BAC JPM MS (only these tickers)
"""
import re
import sys
import warnings
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

DATA = Path("data")
OUT = DATA / "sections"
FLAGS = re.I | re.M


def min_chars(form, name):
    # 10-Q risk factors are often a short "no material changes" blurb
    if form == "10-Q" and name == "risk_factors":
        return 100
    return 5000


# Each entry is (start_pattern, end_pattern) or (start, end, True).
# The optional True means "heading style": the section name is repeated as a page
# header on every page (MS, JPM), so we take the FIRST heading after the previous
# end marker instead of the latest one.
# Everything is anchored to the start of a line so cross-references inside
# sentences don't match.
PATTERNS = {
    "10-K": {
        "risk_factors": [
            (r"^\s*item\s*1a\W{0,6}risk\s*factors",
             r"^\s*item\s*1b\W{0,6}unresolved|^\s*item\s*2\W{0,6}properties"),
            (r"^\s*risk\s*factors\s*$",
             r"^\s*(unresolved\s*staff\s*comments|properties|legal\s*proceedings"
             r"|sustainability|managing\s+global\s+risk"
             r"|management\W?s\s+discussion\s+and\s+analysis)\s*$"),
        ],
        "mdna": [
            (r"^\s*item\s*7\b[\s\S]{0,100}?management\W?s\s+discussion",
             r"^\s*item\s*7a\W{0,6}quantitative|^\s*item\s*8\W{0,6}financial\s+statements"),
            (r"^\s*management\W?s\s+discussion\s+and\s+analysis[^\n]{0,60}$",
             r"^\s*(quantitative\s+and\s+qualitative\s+disclosures?[^\n]{0,60}"
             r"|financial\s+statements\s+and\s+supplementary[^\n]{0,40}"
             r"|risk\s*factors)\s*$",
             True),
            (r"^\s*management\W?s\s+discussion\s+and\s+analysis[^\n]{0,60}$",
             r"^\s*(management\W?s\s+report\s+on\s+internal\s+control[^\n]{0,60}"
             r"|report\s+of\s+independent\s+registered[^\n]{0,60}"
             r"|consolidated\s+financial\s+statements[^\n]{0,40})\s*$",
             True),
        ],
    },
    "10-Q": {
        "risk_factors": [
            (r"^\s*item\s*1a\W{0,6}risk\s*factors",
             r"^\s*item\s*2\W{0,6}unregistered|^\s*item\s*3\W{0,6}defaults"
             r"|^\s*item\s*5\W{0,6}other|^\s*item\s*6\W{0,6}exhibits"),
            (r"^\s*risk\s*factors\s*$",
             r"^\s*(unregistered\s*sales[^\n]{0,60}|other\s+information|exhibits"
             r"|managing\s+global\s+risk|quantitative\s+and\s+qualitative[^\n]{0,60})\s*$"),
        ],
        "mdna": [
            (r"^\s*item\s*2\b[\s\S]{0,100}?management\W?s\s+discussion",
             r"^\s*item\s*3\W{0,6}quantitative|^\s*item\s*4\W{0,6}controls"),
            (r"^\s*management\W?s\s+discussion\s+and\s+analysis[^\n]{0,60}$",
             r"^\s*(quantitative\s+and\s+qualitative\s+disclosures?[^\n]{0,60}"
             r"|controls\s+and\s+procedures|risk\s*factors)\s*$",
             True),
            (r"^\s*management\W?s\s+discussion\s+and\s+analysis[^\n]{0,60}$",
             r"^\s*(consolidated\s+financial\s+statements[^\n]{0,40}"
             r"|report\s+of\s+independent[^\n]{0,60})\s*$",
             True),
        ],
    },
}


def html_to_text(path):
    html = Path(path).read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")
    for t in soup(["script", "style"]):
        t.decompose()
    for t in soup.find_all("ix:header"):  # hidden XBRL junk at top of the file
        t.decompose()
    text = soup.get_text("\n")
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text


def extract(text, start_pat, end_pat, first_start=False):
    starts = [m.start() for m in re.finditer(start_pat, text, FLAGS)]
    ends = [m.start() for m in re.finditer(end_pat, text, FLAGS)]
    best = ""

    if first_start:
        # heading style: for each end marker, take the first heading found in the
        # segment after the previous end marker (page headers repeat after it)
        prev = -1
        for e in ends:
            seg = [s for s in starts if prev < s < e - 10]  # segment after the previous end
            if seg and e - seg[0] > len(best):
                best = text[seg[0]:e]
            prev = e
        return best

    # standard: pair each start with the first end after it. If several starts
    # share the same end, keep the latest start (the real heading, not the
    # table of contents). Then return the longest span.
    pairs = {}
    for s in starts:
        e = next((e for e in ends if e > s + 10), None)
        if e is not None:
            pairs[e] = max(pairs.get(e, -1), s)  # latest start wins
    for e, s in pairs.items():
        if e - s > len(best):
            best = text[s:e]
    return best


def get_section(text, form, name):
    need = min_chars(form, name)
    for spec in PATTERNS[form][name]:
        start_pat, end_pat = spec[0], spec[1]
        first = len(spec) > 2
        sec = extract(text, start_pat, end_pat, first_start=first)
        if len(sec) >= need:
            return sec
    return ""


def main():
    index = pd.read_csv(DATA / "filings_index.csv")
    only = [a.upper() for a in sys.argv[1:]]
    if only:
        index = index[index["ticker"].isin(only)]
    rows = []

    for ticker, grp in index.groupby("ticker"):
        print(f"[{ticker}] parsing {len(grp)} filings...")
        for _, r in grp.iterrows():
            text = html_to_text(r["path"])
            acc = str(r["accessionNumber"]).replace("-", "")
            base = f"{r['form']}_{r['filingDate']}_{acc}"
            out_dir = OUT / ticker
            out_dir.mkdir(parents=True, exist_ok=True)

            if r["form"] == "8-K":
                sections = {"full": text}
            else:
                sections = {
                    name: get_section(text, r["form"], name)
                    for name in PATTERNS[r["form"]]
                }

            for name, sec in sections.items():
                if sec:
                    (out_dir / f"{base}_{name}.txt").write_text(sec, encoding="utf-8")
                rows.append({
                    "ticker": ticker, "form": r["form"],
                    "filingDate": r["filingDate"], "accession": acc,
                    "section": name, "chars": len(sec),
                    "items": r.get("items"),
                })

    new = pd.DataFrame(rows)
    old_path = DATA / "sections_index.csv"
    if only and old_path.exists():
        old = pd.read_csv(old_path)
        sec_index = pd.concat([old[~old["ticker"].isin(only)], new], ignore_index=True)
    else:
        sec_index = new
    sec_index.to_csv(old_path, index=False)

    cov = sec_index[sec_index["form"].isin(["10-K", "10-Q"])].copy()
    cov["found"] = [
        c >= min_chars(f, s) for c, f, s in zip(cov["chars"], cov["form"], cov["section"])
    ]
    found = cov.groupby(["form", "section", "ticker"])["found"].agg(["sum", "count"])
    found["txt"] = found["sum"].astype(int).astype(str) + "/" + found["count"].astype(str)
    print("\nCoverage (found / total):")
    print(found["txt"].unstack("ticker").T)

    med = (
        cov[cov["found"]]
        .groupby(["form", "section", "ticker"])["chars"]
        .median()
        .astype(int)
    )
    print("\nMedian chars of found sections:")
    print(med.unstack("ticker").T)


if __name__ == "__main__":
    main()
