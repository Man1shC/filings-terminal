"""
Step D2: search the filings index (hybrid: embeddings + BM25 keywords).

Used two ways:
  1) from the command line, to check retrieval quality by eye:
       python search.py "commercial real estate credit deterioration" --since 2026-01-01
       python search.py "what drove second quarter results" --ticker GS --since 2026-04-01 --mode dense
  2) as a module by the Q&A agent:   from search import search

Modes:
  dense   embeddings only (finds meaning, misses exact phrases and dates)
  bm25    keywords only   (finds exact terms, misses paraphrases)
  hybrid  both, merged with reciprocal rank fusion  (default)

Filters (all optional): --ticker, --form (10-K | 10-Q | 8-K), --section (risk_factors | mdna |
earnings_release), --since / --until (YYYY-MM-DD, compared to the filing date).

Reads  data/index/chunks.jsonl and data/index/emb.npy  (from build_index.py)
Builds data/index/bm25.pkl the first time hybrid or bm25 is used (about 1-3 minutes, then cached).
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer

MODEL = "BAAI/bge-small-en-v1.5"
BM25_PATH = Path("data/index/bm25.pkl")
K1, B = 1.5, 0.75      # standard BM25 parameters
RRF_K = 60             # reciprocal rank fusion constant
_STATE = {}


def _load():
    if "chunks" not in _STATE:
        _STATE["chunks"] = pd.read_json("data/index/chunks.jsonl", lines=True,
                                        convert_dates=False, dtype={"filingDate": str})
        _STATE["emb"] = np.load("data/index/emb.npy")
        assert len(_STATE["chunks"]) == len(_STATE["emb"]), "chunks and embeddings are out of sync"
    return _STATE["chunks"], _STATE["emb"]


def _hasher():
    # hashing keeps memory small (no giant vocabulary) while still covering two-word phrases
    return HashingVectorizer(n_features=2 ** 20, ngram_range=(1, 2), alternate_sign=False,
                             norm=None, stop_words="english", lowercase=True)


def _bm25():
    """(hasher, BM25 weight matrix in CSC form). Built once, cached on disk."""
    if "bm25" not in _STATE:
        chunks, _ = _load()
        hv = _hasher()
        if BM25_PATH.exists():
            with open(BM25_PATH, "rb") as f:
                W = pickle.load(f)
            if W.shape[0] != len(chunks):  # index was rebuilt: cache is stale
                W = None
        else:
            W = None
        if W is None:
            print("Building the keyword index (one time, 1-3 minutes)...", flush=True)
            X = hv.transform(chunks["text"]).tocsr()
            n = X.shape[0]
            df = np.bincount(X.indices, minlength=X.shape[1])           # docs containing each term
            idf = np.log(1 + (n - df + 0.5) / (df + 0.5))
            dl = np.asarray(X.sum(axis=1)).ravel()                      # chunk length in terms
            rows = np.repeat(np.arange(n), np.diff(X.indptr))
            tf = X.data.astype(np.float64)
            X.data = tf * (K1 + 1) / (tf + K1 * (1 - B + B * dl[rows] / dl.mean())) * idf[X.indices]
            W = X.tocsc()
            with open(BM25_PATH, "wb") as f:
                pickle.dump(W, f)
        _STATE["bm25"] = (hv, W)
    return _STATE["bm25"]


def bm25_scores(query):
    """BM25 score of every chunk for a query."""
    hv, W = _bm25()
    cols = np.unique(hv.transform([query]).indices)
    if len(cols) == 0:
        return np.zeros(W.shape[0])
    return np.asarray(W[:, cols].sum(axis=1)).ravel()


def embed_query(q):
    """Unit-length embedding of a question (bge models use a special query prefix)."""
    if "model" not in _STATE:
        from fastembed import TextEmbedding
        _STATE["model"] = TextEmbedding(MODEL)
    model = _STATE["model"]
    gen = model.query_embed([q]) if hasattr(model, "query_embed") else model.embed([q])
    v = np.array(list(gen)[0], dtype=np.float32)
    return v / np.linalg.norm(v)


def _ranks(scores):
    """1-based rank of each item (1 = best)."""
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.int64)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks


def search(query, k=5, ticker=None, form=None, section=None, since=None, until=None,
           mode="hybrid"):
    """Top-k chunks for a question: DataFrame with 'score' (what it is sorted by),
    plus 'dense' (cosine similarity) and 'bm25' (keyword score)."""
    chunks, emb = _load()
    mask = np.ones(len(chunks), dtype=bool)
    if ticker:
        mask &= (chunks["ticker"] == ticker.upper()).to_numpy()
    if form:
        mask &= (chunks["form"] == form).to_numpy()
    if section:
        mask &= (chunks["section"] == section).to_numpy()
    if since:
        mask &= (chunks["filingDate"] >= since).to_numpy()
    if until:
        mask &= (chunks["filingDate"] <= until).to_numpy()
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return chunks.iloc[0:0].assign(score=[], dense=[], bm25=[])

    dense = emb[idx] @ embed_query(query) if mode in ("dense", "hybrid") else np.zeros(len(idx))
    kw = bm25_scores(query)[idx] if mode in ("bm25", "hybrid") else np.zeros(len(idx))

    if mode == "dense":
        score = dense
    elif mode == "bm25":
        score = kw
    else:  # hybrid: reciprocal rank fusion; a chunk with no keyword match gets nothing from that side
        score = 1.0 / (RRF_K + _ranks(dense)) + np.where(kw > 0, 1.0 / (RRF_K + _ranks(kw)), 0.0)

    order = np.argsort(-score, kind="stable")[:k]
    out = chunks.iloc[idx[order]].copy()
    out["score"], out["dense"], out["bm25"] = score[order], dense[order], kw[order]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--ticker")
    ap.add_argument("--form")
    ap.add_argument("--section")
    ap.add_argument("--since")
    ap.add_argument("--until")
    ap.add_argument("--mode", choices=["dense", "bm25", "hybrid"], default="hybrid")
    a = ap.parse_args()

    res = search(a.query, a.k, a.ticker, a.form, a.section, a.since, a.until, a.mode)
    if res.empty:
        print("No chunks match those filters.")
        return
    print(f"mode = {a.mode}")
    for _, r in res.iterrows():
        print(f"\n[{r['id']}] {r['ticker']} {r['form']} {r['section']} {r['filingDate']}  "
              f"score={r['score']:.4f}  (dense={r['dense']:.3f}, keyword={r['bm25']:.1f})")
        print("   " + r["text"][:450].replace("\n", " ") + (" ..." if len(r["text"]) > 450 else ""))


if __name__ == "__main__":
    main()
