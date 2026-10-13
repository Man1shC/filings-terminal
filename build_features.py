"""
Step 3: build trend features from the parsed sections.

Reads  data/sections_index.csv + data/sections/<TICKER>/*.txt   (from step 2)
Writes data/features.csv

One row per (ticker, form, section, filing) with:
  chars, words                     size of the section (0 = section missing)
  chars_change                     change vs the previous filing of the same kind
  neg_rate / unc_rate / pos_rate   share of negative / uncertain / positive words per 1,000 words
  net_tone                         pos_rate - neg_rate
  drift                            1 - TF-IDF cosine similarity vs the previous filing's text
                                   (0 = copy-paste, higher = more new language)

Run:  python build_features.py
"""
import re
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA = Path("data")
SEC = DATA / "sections"

# companies whose 10-K Risk Factors, 10-K MD&A and 10-Q MD&A all parsed cleanly
KEEP = [
    "AXP", "BAC", "C", "CFG", "COF", "EVR", "FITB", "GS", "HBAN", "HLI", "JEF",
    "JPM", "KEY", "LAZ", "MS", "MTB", "RF", "RJF", "SCHW", "STT", "TFC",
]

# Starter word lists (finance-flavoured). Later we can swap in the
# Loughran-McDonald finance dictionary, the standard in this field.
NEGATIVE = {
    "loss", "losses", "decline", "declined", "declining", "adverse", "adversely",
    "deterioration", "deteriorated", "impairment", "impaired", "default", "defaults",
    "downturn", "weakness", "weakened", "litigation", "penalties", "fines", "failure",
    "failed", "negative", "unfavorable", "disruption", "stress", "stressed", "shortfall",
}
UNCERTAIN = {
    "could", "may", "might", "uncertain", "uncertainty", "uncertainties",
    "unpredictable", "possible", "possibly", "volatile", "volatility", "unclear",
    "unknown", "fluctuate", "fluctuations", "depend", "depends", "dependent",
}
POSITIVE = {
    "strong", "strength", "strengthened", "growth", "improved", "improvement", "gain",
    "gains", "favorable", "profitable", "record", "resilient", "robust", "benefit",
    "benefited", "outperformed", "momentum", "successful", "opportunity", "opportunities",
}


def read_text(row):
    path = SEC / row["ticker"] / (
        f"{row['form']}_{row['filingDate']}_{row['accession']}_{row['section']}.txt"
    )
    return path.read_text(encoding="utf-8") if path.exists() else ""


def tone_features(text):
    words = re.findall(r"[a-z]+", text.lower())
    n = len(words)
    if n == 0:
        return {"words": 0, "neg_rate": None, "unc_rate": None, "pos_rate": None}
    per_k = 1000 / n
    return {
        "words": n,
        "neg_rate": sum(w in NEGATIVE for w in words) * per_k,
        "unc_rate": sum(w in UNCERTAIN for w in words) * per_k,
        "pos_rate": sum(w in POSITIVE for w in words) * per_k,
    }


def add_drift(g):
    """TF-IDF similarity between each filing and the previous non-empty one."""
    g = g.sort_values("filingDate").copy()
    texts = g["text"].tolist()
    ok = [i for i, t in enumerate(texts) if len(t) > 0]
    drift = [None] * len(g)
    if len(ok) >= 2:
        # strip company names (JPMorgan Chase -> JPMorganChase broke drift) and numbers
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=50000, sublinear_tf=True, token_pattern=r"(?u)\b[a-zA-Z]{2,}\b", preprocessor=lambda s: re.sub(r"jpmorgan\s*chase|jpmorgan|bank of america|goldman sachs|morgan stanley|citigroup|citibank|\bciti\b", " ", s.lower()))
        X = vec.fit_transform([texts[i] for i in ok])
        for a in range(len(ok) - 1):
            sim = cosine_similarity(X[a], X[a + 1])[0, 0]
            drift[ok[a + 1]] = 1 - sim
    g["drift"] = drift
    return g


def main():
    idx = pd.read_csv(DATA / "sections_index.csv")
    idx = idx[idx["form"].isin(["10-K", "10-Q"]) & idx["ticker"].isin(KEEP)].copy()
    # pandas reads accession as a number and drops leading zeros; restore to 18 digits
    idx["accession"] = idx["accession"].astype(str).str.zfill(18)

    print(f"Reading {len(idx)} sections...")
    idx["text"] = [read_text(r) for _, r in idx.iterrows()]

    tone = pd.DataFrame([tone_features(t) for t in idx["text"]], index=idx.index)
    idx = pd.concat([idx, tone], axis=1)
    idx["net_tone"] = idx["pos_rate"] - idx["neg_rate"]

    print("Computing drift (TF-IDF)...")
    parts = [add_drift(g) for _, g in idx.groupby(["ticker", "form", "section"])]
    df = pd.concat(parts).sort_values(["ticker", "form", "section", "filingDate"])

    df["chars_change"] = df.groupby(["ticker", "form", "section"])["chars"].diff()
    df = df.drop(columns=["text", "items"], errors="ignore")
    df.to_csv(DATA / "features.csv", index=False)
    print(f"Saved {len(df)} rows -> data/features.csv\n")

    df["year"] = df["filingDate"].str[:4]
    k = df[(df["form"] == "10-K") & (df["section"] == "risk_factors")]
    print("10-K Risk Factors drift by filing year (higher = more new language):")
    print(k.pivot(index="ticker", columns="year", values="drift").round(3))

    m = df[(df["form"] == "10-K") & (df["section"] == "mdna")]
    print("\n10-K MD&A net tone by filing year (positive - negative per 1,000 words):")
    print(m.pivot(index="ticker", columns="year", values="net_tone").round(2))


if __name__ == "__main__":
    main()
