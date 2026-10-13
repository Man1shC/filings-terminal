"""
Step D6: evaluate the Q&A agent's ANSWERS.

For each sampled eval question:
  1. the agent answers it (with tools)                     -> agent_answer, citations
  2. the same model answers it with NO documents           -> baseline_answer
  3. a judge model grades both against the reference answer   correct / partial / incorrect / no_answer
  4. the judge lists the agent answer's factual claims and checks each against ONLY the
     passages cited next to it                             -> faithfulness
  5. citations are checked against what was actually retrieved

Run:  python eval_agent.py 10      (test: 10 questions)
      python eval_agent.py 60      (60 questions; resumes, and the first 10 are kept)

Needs: ANTHROPIC_API_KEY in this PowerShell window.  Writes data/eval/agent_results.jsonl
The judge is a different, stronger model than the agent to limit self-grading bias.
"""
import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

import agent as A

JUDGE_MODEL = "claude-opus-5-5"
OUT = Path("data/eval/agent_results.jsonl")
WORKERS = 3
GRADES = {"correct", "partial", "incorrect", "no_answer"}
VERDICTS = {"supported", "unsupported", "contradicted"}

BASELINE_SYSTEM = ("Answer the question from your own knowledge only. If you do not know, say so briefly. "
                   "Do not invent figures.")

GRADE_SYSTEM = "You grade answers to questions about SEC filings. Reply with a single JSON object and nothing else."
GRADE_USER = """QUESTION:
{question}

REFERENCE ANSWER (taken from the filing):
{reference}

CANDIDATE ANSWER:
{answer}

Grade the candidate against the reference:
- "correct": contains the key facts of the reference and nothing that contradicts it
- "partial": contains some of the key facts but misses important ones
- "incorrect": wrong, contradicts the reference, or states specific facts the reference does not support
- "no_answer": says it cannot find the information or does not know

Return JSON: {{"grade": "<one of the four>", "reason": "<at most 25 words>"}}"""

FAITH_SYSTEM = "You check whether an answer is supported by the passages it cites. Reply with a single JSON object and nothing else."
FAITH_USER = """ANSWER (citations appear in square brackets after the claims they support):
{answer}

CITED PASSAGES (the ONLY evidence you may use):
{sources}

List the factual claims in the answer (numbers, events, statements about the company; skip pure
connecting words and statements of what the answer itself covers). For each claim decide, using ONLY
the passages cited next to that claim:
- "supported": a cited passage states it
- "unsupported": the cited passages do not state it (even if it may be true)
- "contradicted": a cited passage says something different

Return JSON: {{"claims": [{{"claim": "<short>", "verdict": "<supported|unsupported|contradicted>"}}]}}"""


def parse_json(raw):
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        raise ValueError(f"no JSON in reply: {raw[:200]!r}")
    return json.loads(m.group(0))


def text_of(resp):
    return "".join(b.text for b in resp.content if b.type == "text")


def judge(client, model, system, user):
    resp = client.messages.create(model=model, max_tokens=3000, system=system,
                                  messages=[{"role": "user", "content": user}])
    return parse_json(text_of(resp))


def grade(client, model, row, answer):
    g = judge(client, model, GRADE_SYSTEM, GRADE_USER.format(
        question=row["question"], reference=row["reference_answer"], answer=answer))
    if g.get("grade") not in GRADES:
        raise ValueError(f"bad grade: {g}")
    return g["grade"], str(g.get("reason", ""))[:200]


def evaluate_question(client, row, agent_model, judge_model):
    r = A.ask(row["question"], agent_model, client=client)
    base = text_of(client.messages.create(
        model=agent_model, max_tokens=800, system=BASELINE_SYSTEM,
        messages=[{"role": "user", "content": row["question"]}]))

    agent_grade, agent_reason = grade(client, judge_model, row, r["answer"])
    base_grade, base_reason = grade(client, judge_model, row, base)

    cited = [c for c in r["cited"] if c in r["sources"]]
    sources = "\n\n".join(
        f"[{c}] {r['sources'][c]['label']}\n{r['sources'][c].get('text', '(no text stored)')[:1600]}"
        for c in cited) or "(the answer cites no sources)"
    f = judge(client, judge_model, FAITH_SYSTEM, FAITH_USER.format(answer=r["answer"], sources=sources))
    claims = [c for c in f.get("claims", []) if c.get("verdict") in VERDICTS]

    return {
        "chunk_id": int(row["chunk_id"]), "ticker": row["ticker"], "form": row["form"],
        "section": row["section"], "question": row["question"],
        "reference_answer": row["reference_answer"],
        "agent_answer": r["answer"], "baseline_answer": base,
        "agent_grade": agent_grade, "agent_reason": agent_reason,
        "baseline_grade": base_grade, "baseline_reason": base_reason,
        "n_claims": len(claims),
        "n_supported": sum(c["verdict"] == "supported" for c in claims),
        "n_contradicted": sum(c["verdict"] == "contradicted" for c in claims),
        "unsupported_claims": [c["claim"] for c in claims if c["verdict"] != "supported"][:6],
        "n_cited": len(r["cited"]), "invalid_citations": r["invalid_citations"],
        "n_tool_calls": len(r["tool_calls"]),
    }


def report(df):
    n = len(df)
    print(f"\n=== RESULTS on {n} questions (graded against reference answers by {JUDGE_MODEL}) ===\n")
    tab = pd.DataFrame({
        "agent": df["agent_grade"].value_counts(normalize=True),
        "no-documents baseline": df["baseline_grade"].value_counts(normalize=True),
    }).reindex(["correct", "partial", "incorrect", "no_answer"]).fillna(0)
    print((tab * 100).round(1).T.to_string(), "  (percent of questions)\n")

    has = df[df["n_claims"] > 0]
    per_answer = has["n_supported"] / has["n_claims"]
    print(f"Faithfulness (claims supported by the passages cited next to them): "
          f"{has['n_supported'].sum() / has['n_claims'].sum():.1%} of {int(has['n_claims'].sum())} claims")
    print(f"Answers where EVERY claim is supported: {(per_answer == 1).mean():.1%} of {len(has)} answers")
    print(f"Claims contradicted by their cited passage: {int(df['n_contradicted'].sum())}")
    bad = int((df["invalid_citations"].map(len) > 0).sum())
    print(f"Answers citing a source that was never retrieved: {bad} of {n}")
    print(f"Answers with no citations at all: {int((df['n_cited'] == 0).sum())} of {n}")
    print(f"Average tool calls per answer: {df['n_tool_calls'].mean():.1f}\n")

    print("Agent correctness by section (percent 'correct'):")
    print((df.groupby("section")["agent_grade"].apply(lambda s: (s == "correct").mean() * 100)
           .round(1).to_string()), "\n")

    wrong = df[df["agent_grade"].isin(["incorrect", "no_answer"])].head(5)
    if len(wrong):
        print("A few answers graded incorrect / no_answer (for debugging):")
        for _, r in wrong.iterrows():
            print(f"  - [{r['ticker']} {r['form']} {r['section']}] {r['question'][:110]}\n"
                  f"      judge: {r['agent_reason']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int, nargs="?", default=10)
    ap.add_argument("--agent-model", default=A.MODEL)
    ap.add_argument("--judge-model", default=JUDGE_MODEL)
    a = ap.parse_args()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit('Set your key first, in PowerShell:  $env:ANTHROPIC_API_KEY = "your-key"')
    import anthropic
    client = anthropic.Anthropic(max_retries=5)

    rows = [json.loads(l) for l in open("data/eval/questions.jsonl", encoding="utf-8") if l.strip()]
    qs = pd.DataFrame([r for r in rows if not r["skip"]]).sample(frac=1, random_state=0)

    done = set()
    if OUT.exists():
        done = {json.loads(l)["chunk_id"] for l in OUT.read_text(encoding="utf-8").splitlines() if l.strip()}
    todo = qs[~qs["chunk_id"].isin(done)].head(max(0, a.n - len(done)))
    print(f"{len(done)} already done; running {len(todo)} more "
          f"(agent: {a.agent_model}, judge: {a.judge_model})")

    failed = 0
    with ThreadPoolExecutor(WORKERS) as ex:
        futs = {ex.submit(evaluate_question, client, r, a.agent_model, a.judge_model): r
                for _, r in todo.iterrows()}
        for i, fut in enumerate(as_completed(futs), 1):
            r = futs[fut]
            try:
                res = fut.result()
            except Exception as e:
                failed += 1
                print(f"[error] chunk {r['chunk_id']} ({r['ticker']}): {type(e).__name__}: {e}")
                continue
            with OUT.open("a", encoding="utf-8") as f:
                f.write(json.dumps(res) + "\n")
            print(f"  {i}/{len(todo)} done (agent: {res['agent_grade']}, baseline: {res['baseline_grade']})",
                  flush=True)

    if OUT.exists():
        report(pd.DataFrame([json.loads(l) for l in OUT.read_text(encoding="utf-8").splitlines() if l.strip()]))
    if failed:
        print(f"\n{failed} questions failed; re-run the same command to retry them.")


if __name__ == "__main__":
    main()
