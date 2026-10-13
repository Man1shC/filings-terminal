"""
Step C1: extract structured signals from earnings releases with Claude.

Each release becomes one row of facts: results vs last year, guidance direction,
credit-quality direction, capital return, one-time items, tone (1-5), main driver.
Those become new features for the same event-study harness we already built.

Run:  python llm_extract.py 5     (test on 5 random releases: prints results + token use)
      python llm_extract.py       (all releases; resumes where it left off if interrupted)

Needs:  pip install anthropic
        PowerShell:  $env:ANTHROPIC_API_KEY = "your-key"   (this terminal only;
                     never paste the key into this file or into chat)

Writes data/llm_extractions.jsonl (one JSON object per line)
"""
import json
import os
import re
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

MODEL = "claude-haiku-5-5"  # cheap and fast; fine for structured extraction
MAX_CHARS = 30000           # the narrative is at the top; tables further down add cost, not signal
WORKERS = 4
OUT = Path("data/llm_extractions.jsonl")

SYSTEM = (
    "You are a financial analyst extracting structured facts from a company's earnings "
    "press release. Use ONLY the text provided. Do not use outside knowledge about the "
    "company, its stock price reaction, or any later events. Reply with a single JSON "
    "object and nothing else."
)

PROMPT = """Extract these fields from the earnings press release below.

- results_vs_prior_year: "better" | "similar" | "worse"
  (headline net income / EPS versus the same quarter a year earlier)
- guidance: "raised" | "lowered" | "maintained" | "none"
  (any forward outlook or targets, e.g. net interest income, expenses, revenue, margins;
   "none" if the release gives no outlook)
- credit_quality: "improving" | "stable" | "deteriorating" | "not_discussed"
  (provisions, charge-offs, delinquencies, or credit commentary; "not_discussed" if absent)
- capital_return: true | false
  (a NEW or INCREASED buyback authorization or dividend is announced in this release)
- notable_items: "positive" | "negative" | "mixed" | "none"
  (one-time or significant items that change reported results)
- tone: integer 1-5 (1 = very negative, 3 = neutral, 5 = very positive),
  judged from management's own language about the quarter
- driver: string of at most 15 words naming the main reason for the result

Return JSON with exactly these keys."""

ALLOWED = {
    "results_vs_prior_year": {"better", "similar", "worse"},
    "guidance": {"raised", "lowered", "maintained", "none"},
    "credit_quality": {"improving", "stable", "deteriorating", "not_discussed"},
    "notable_items": {"positive", "negative", "mixed", "none"},
}


def release_text(path):
    html = Path(path).read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")
    for t in soup(["script", "style"]):
        t.decompose()
    return " ".join(soup.get_text(" ").split())


def parse_json(raw):
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        raise ValueError(f"no JSON in reply: {raw[:200]!r}")
    return json.loads(m.group(0))


def validate(d):
    for key, allowed in ALLOWED.items():
        if d.get(key) not in allowed:
            raise ValueError(f"bad value for {key}: {d.get(key)!r}")
    if not isinstance(d.get("capital_return"), bool):
        raise ValueError(f"bad value for capital_return: {d.get('capital_return')!r}")
    if d.get("tone") not in (1, 2, 3, 4, 5):
        raise ValueError(f"bad value for tone: {d.get('tone')!r}")
    d["driver"] = str(d.get("driver", ""))[:200]
    return {k: d[k] for k in [*ALLOWED, "capital_return", "tone", "driver"]}


def extract(client, row):
    text = release_text(row["path"])[:MAX_CHARS]
    msg = client.messages.create(
        model=MODEL,
        max_tokens=2000,  # room in case the model reasons before answering
        system=SYSTEM,
        messages=[{"role": "user", "content": PROMPT + "\n\nPRESS RELEASE:\n" + text}],
    )
    raw = "".join(b.text for b in msg.content if b.type == "text")
    d = validate(parse_json(raw))
    return {
        "ticker": row["ticker"], "filingDate": row["filingDate"],
        "accession": row["accession"], **d,
        "in_tokens": msg.usage.input_tokens, "out_tokens": msg.usage.output_tokens,
    }


def main():
    import anthropic

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit('Set your key first, in PowerShell:  $env:ANTHROPIC_API_KEY = ""')
    client = anthropic.Anthropic(max_retries=5)

    idx = pd.read_csv("data/earnings_index.csv", dtype={"accession": str})
    done = set()
    if OUT.exists():
        done = {json.loads(line)["accession"] for line in OUT.read_text(encoding="utf-8").splitlines()
                if line.strip()}
    todo = idx[~idx["accession"].isin(done)]
    test = len(sys.argv) > 1
    if test:
        todo = todo.sample(min(int(sys.argv[1]), len(todo)), random_state=0)
    print(f"{len(done)} already done, {len(todo)} to process with {MODEL}")

    results = []
    with ThreadPoolExecutor(WORKERS) as ex:
        futs = {ex.submit(extract, client, r): r for _, r in todo.iterrows()}
        for i, fut in enumerate(as_completed(futs), 1):
            r = futs[fut]
            try:
                res = fut.result()
            except Exception as e:
                print(f"[error] {r['ticker']} {r['filingDate']}: {e}")
                continue
            with OUT.open("a", encoding="utf-8") as f:
                f.write(json.dumps(res) + "\n")
            results.append(res)
            if not test and i % 25 == 0:
                print(f"  {i}/{len(todo)} done")

    if test and results:
        for r in sorted(results, key=lambda x: (x["ticker"], x["filingDate"])):
            print(f"\n{r['ticker']} {r['filingDate']}  vs PY: {r['results_vs_prior_year']} | "
                  f"guidance: {r['guidance']} | credit: {r['credit_quality']} | "
                  f"capital return: {r['capital_return']} | items: {r['notable_items']} | "
                  f"tone: {r['tone']}\n   driver: {r['driver']}")
        tin = sum(r["in_tokens"] for r in results) / len(results)
        tout = sum(r["out_tokens"] for r in results) / len(results)
        print(f"\nAverage per release: {tin:.0f} input tokens, {tout:.0f} output tokens.")
        print(f"Full run of {len(idx)} releases is about {tin * len(idx) / 1e6:.1f}M input tokens "
              f"and {tout * len(idx) / 1e6:.2f}M output tokens.")
    print(f"\nDone: {len(results)} extracted this run, {len(todo) - len(results)} failed.")


if __name__ == "__main__":
    main()
