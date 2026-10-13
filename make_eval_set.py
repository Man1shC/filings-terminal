"""
Step D3: generate an evaluation set for the retrieval system and the Q&A agent.

Samples passages from the index (35% risk factors, 35% MD&A, 30% earnings releases),
and asks Claude to write ONE question each passage answers, naming the company and the
filing period, plus a short reference answer. Passages that are table fragments or
boilerplate are skipped by the model.

Run:  python make_eval_set.py 5     (test: 5 passages, questions printed)
      python make_eval_set.py       (full: about 200 passages; resumes if interrupted)

Needs:  ANTHROPIC_API_KEY set in this PowerShell window
Writes: data/eval/questions.jsonl   (one JSON object per sampled passage; skipped ones have skip=true)
"""
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

MODEL = "claude-haiku-5-5"
N_CANDIDATES = 210
WEIGHTS = {"risk_factors": 0.35, "mdna": 0.35, "earnings_release": 0.30}
WORKERS = 4
OUT = Path("data/eval/questions.jsonl")

SYSTEM = (
    "You write evaluation questions for a retrieval system over SEC filings. "
    "Reply with a single JSON object and nothing else."
)

PROMPT = """Below is one passage from a company's SEC filing, with its metadata.

Company ticker: {ticker}
Document: {form} {section_name}, filed {month_year}

Write ONE question that this passage answers, the way a financial analyst might ask it. Rules:
- Name the company by its ticker and refer to the filing by form and filing month and year
  (for example "in JPM's 10-K filed in February 2026"). Do not guess fiscal years.
- A reader who has NOT seen the passage must be able to tell which document the question is about.
- Do not copy phrases from the passage word for word; paraphrase.
- The reference answer must be 1-2 sentences and use ONLY facts stated in the passage.
- If the passage is mostly table fragments, boilerplate, a cross-reference, or has no
  standalone substance, set "skip" to true.

Return JSON with exactly these keys: "skip" (true/false), "question" (string), "reference_answer" (string).

PASSAGE:
{text}"""

SECTION_NAMES = {"risk_factors": "Risk Factors section", "mdna": "MD&A section",
                 "earnings_release": "earnings press release"}


def parse_json(raw):
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        raise ValueError(f"no JSON in reply: {raw[:200]!r}")
    return json.loads(m.group(0))


def candidates():
    chunks = pd.read_json("data/index/chunks.jsonl", lines=True, convert_dates=False,
                          dtype={"filingDate": str})
    alpha = chunks["text"].map(lambda t: sum(c.isalpha() for c in t) / len(t))
    pool = chunks[(chunks["text"].str.len() >= 700) & (alpha >= 0.75)]
    parts = []
    for sec, w in WEIGHTS.items():
        sub = pool[pool["section"] == sec]
        parts.append(sub.sample(min(int(N_CANDIDATES * w), len(sub)), random_state=0))
    return pd.concat(parts).sample(frac=1, random_state=0)


def make_question(client, row):
    date = pd.Timestamp(row["filingDate"])
    msg = client.messages.create(
        model=MODEL,
        max_tokens=1500,
        system=SYSTEM,
        messages=[{"role": "user", "content": PROMPT.format(
            ticker=row["ticker"], form=row["form"],
            section_name=SECTION_NAMES[row["section"]],
            month_year=date.strftime("%B %Y"), text=row["text"])}],
    )
    d = parse_json("".join(b.text for b in msg.content if b.type == "text"))
    base = {"chunk_id": int(row["id"]), "ticker": row["ticker"], "form": row["form"],
            "section": row["section"], "filingDate": row["filingDate"]}
    if d.get("skip") is True:
        return {**base, "skip": True}
    q, a = d.get("question"), d.get("reference_answer")
    if not (isinstance(q, str) and len(q) >= 20 and isinstance(a, str) and len(a) >= 10):
        raise ValueError(f"bad question or answer: {d}")
    return {**base, "skip": False, "question": q.strip(), "reference_answer": a.strip()}


def main():
    import anthropic

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit('Set your key first, in PowerShell:  $env:ANTHROPIC_API_KEY = "your-key"')
    client = anthropic.Anthropic(max_retries=5)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if OUT.exists():
        done = {json.loads(l)["chunk_id"] for l in OUT.read_text(encoding="utf-8").splitlines() if l.strip()}
    todo = candidates()
    todo = todo[~todo["id"].isin(done)]
    test = len(sys.argv) > 1
    if test:
        todo = todo.head(int(sys.argv[1]))
    print(f"{len(done)} already done, {len(todo)} to process with {MODEL}")

    made, skipped, failed = [], 0, 0
    with ThreadPoolExecutor(WORKERS) as ex:
        futs = {ex.submit(make_question, client, r): r for _, r in todo.iterrows()}
        for i, fut in enumerate(as_completed(futs), 1):
            r = futs[fut]
            try:
                res = fut.result()
            except Exception as e:
                failed += 1
                print(f"[error] chunk {r['id']} ({r['ticker']}): {e}")
                continue
            with OUT.open("a", encoding="utf-8") as f:
                f.write(json.dumps(res) + "\n")
            if res["skip"]:
                skipped += 1
            else:
                made.append(res)
            if not test and i % 25 == 0:
                print(f"  {i}/{len(todo)} done")

    if test:
        for r in made:
            print(f"\n[{r['ticker']} {r['form']} {r['section']} {r['filingDate']}, chunk {r['chunk_id']}]")
            print(f"  Q: {r['question']}\n  A: {r['reference_answer']}")
    print(f"\nDone: {len(made)} questions, {skipped} skipped as unsuitable, {failed} failed.")


if __name__ == "__main__":
    main()
