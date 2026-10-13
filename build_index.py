"""
Step D1: build the search index for the filings Q&A agent.

What it indexes (21 companies, filings dated SINCE or later):
  - 10-K and 10-Q  Risk Factors and MD&A sections   (data/sections/<TICKER>/*.txt)
  - earnings press releases (first 40k chars)        (data/earnings/<TICKER>/*.htm)

Each document is cut into ~1,500-character chunks (paragraph-aware, small overlap),
table-dump chunks are dropped, and every chunk is embedded locally (no API cost).

Writes  data/index/chunks.jsonl   one row per chunk: id, ticker, form, section, filingDate, text
        data/index/emb.npy        unit-length embeddings, row i = chunk id i
Progress is printed every 500 chunks with speed and time left.
Safe to interrupt and re-run: finished 2,000-chunk parts are kept.
Documents are ordered newest-first, so lowering SINCE later only ADDS older parts at the end.

Needs:  pip install fastembed
Run:    python build_index.py
"""
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

KEEP = [
    "AXP", "BAC", "C", "CFG", "COF", "EVR", "FITB", "GS", "HBAN", "HLI", "JEF",
    "JPM", "KEY", "LAZ", "MS", "MTB", "RF", "RJF", "SCHW", "STT", "TFC",
]
SINCE = "2025-01-01"  # only index filings on or after this date (lower it to index more)
MODEL = "BAAI/bge-small-en-v1.5"
CHUNK = 1500          # target characters per chunk
MIN_CHUNK = 200       # drop tiny fragments
MIN_ALPHA = 0.6       # drop table dumps (mostly digits and symbols)
RELEASE_CHARS = 40000 # earnings releases: narrative is at the top
SHARD = 2000          # chunks per saved part
SUB = 500             # chunks per progress update

INDEX = Path("data/index")
FILE_RE = re.compile(r"^(10-K|10-Q)_(\d{4}-\d{2}-\d{2})_(\d+)_(risk_factors|mdna)\.txt$")


def chunk_text(text, size=CHUNK):
    """Paragraph-aware chunks of about `size` characters, with the previous paragraph as overlap."""
    units = []
    for para in (p.strip() for p in text.split("\n")):
        if not para:
            continue
        if len(para) <= size:
            units.append(para)
            continue
        cur = ""  # very long paragraph: pack its sentences
        for sent in re.split(r"(?<=[.!?])\s+", para):
            if cur and len(cur) + len(sent) + 1 > size:
                units.append(cur)
                cur = ""
            cur = (cur + " " + sent).strip()
        if cur:
            units.append(cur)

    chunks, cur, last = [], "", ""
    for u in units:
        if cur and len(cur) + len(u) + 1 > size:
            chunks.append(cur)
            cur = last if 0 < len(last) <= 300 else ""  # small overlap
        cur = (cur + "\n" + u).strip()
        last = u
    if cur:
        chunks.append(cur)
    return chunks


def good_chunk(c):
    if len(c) < MIN_CHUNK:
        return False
    return sum(ch.isalpha() for ch in c) / len(c) >= MIN_ALPHA


def release_text(path):
    html = Path(path).read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")
    for t in soup(["script", "style"]):
        t.decompose()
    lines = [" ".join(x.split()) for x in soup.get_text("\n").split("\n")]
    return "\n".join(x for x in lines if x)


def documents():
    """Yield (meta, text) for every document filed on or after SINCE, newest first."""
    items = []
    for t in KEEP:
        for p in Path(f"data/sections/{t}").glob("*.txt"):
            m = FILE_RE.match(p.name)
            if m and m.group(2) >= SINCE:
                form, date, _, section = m.groups()
                items.append(({"ticker": t, "form": form, "section": section,
                               "filingDate": date}, p, "section"))
        for p in Path(f"data/earnings/{t}").glob("*.htm"):
            date = p.name.split("_")[0]
            if date >= SINCE:
                items.append(({"ticker": t, "form": "8-K", "section": "earnings_release",
                               "filingDate": date}, p, "release"))
    # newest first, then a stable order: older filings can be added later without redoing parts
    items.sort(key=lambda x: (x[0]["filingDate"], x[0]["ticker"], x[0]["form"], x[0]["section"]),
               reverse=True)
    for meta, p, kind in items:
        text = p.read_text(encoding="utf-8") if kind == "section" else release_text(p)[:RELEASE_CHARS]
        yield meta, text


def embed_texts(texts):
    """Unit-length embeddings for a list of texts (local model, no API)."""
    from fastembed import TextEmbedding
    global _MODEL
    if "_MODEL" not in globals():
        _MODEL = TextEmbedding(MODEL)
    vecs = np.array(list(_MODEL.embed(texts, batch_size=64)), dtype=np.float32)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def main():
    INDEX.mkdir(parents=True, exist_ok=True)

    rows = []
    for meta, text in documents():
        for c in chunk_text(text):
            if good_chunk(c):
                rows.append({**meta, "text": c})
    chunks = pd.DataFrame(rows)
    chunks.insert(0, "id", range(len(chunks)))
    chunks.to_json(INDEX / "chunks.jsonl", orient="records", lines=True)

    total = len(chunks)
    print(f"{total} chunks from {chunks.groupby(['ticker','form','section','filingDate']).ngroups} "
          f"documents (filed {SINCE} or later)")
    print(chunks.groupby(["form", "section"]).size().to_string(), "\n")

    n_parts = (total + SHARD - 1) // SHARD
    t0, processed = time.time(), 0
    for s in range(n_parts):
        lo, hi = s * SHARD, min((s + 1) * SHARD, total)
        f = INDEX / f"part_{s:04d}.npy"
        if f.exists() and np.load(f).shape[0] == hi - lo:
            print(f"part {s + 1}/{n_parts}: already done")
            continue
        texts = chunks["text"].iloc[lo:hi].tolist()
        vecs = []
        for j in range(0, len(texts), SUB):
            vecs.append(embed_texts(texts[j:j + SUB]))
            processed += len(texts[j:j + SUB])
            rate = processed / (time.time() - t0)
            left = total - (lo + j + len(texts[j:j + SUB]))
            print(f"  part {s + 1}/{n_parts}: {lo + j + len(texts[j:j + SUB])}/{total} chunks | "
                  f"{rate:.1f} chunks/s | about {left / rate / 60:.0f} min left", flush=True)
        np.save(f, np.vstack(vecs))

    emb = np.vstack([np.load(INDEX / f"part_{s:04d}.npy") for s in range(n_parts)])
    assert len(emb) == total
    np.save(INDEX / "emb.npy", emb)
    print(f"\nSaved data/index/emb.npy {emb.shape} and data/index/chunks.jsonl")


if __name__ == "__main__":
    main()
