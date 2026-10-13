"""
Step D4: measure retrieval quality on the eval questions.

For every question, search with each mode (dense, bm25, hybrid) in two settings:
  unfiltered  the question text only
  filtered    + company, form and filing month, which is what the agent sets from the question

Metrics (gold = the passage the question was written from):
  hit@1 / hit@5 / hit@10   share of questions where the gold passage is in the top 1 / 5 / 10
  MRR                      mean reciprocal rank of the gold passage (0 if not in the top 10)
  doc_hit@5                share where ANY chunk of the gold document is in the top 5
                           (right filing, even if a neighbouring passage)

Reads  data/eval/questions.jsonl  (make_eval_set.py)
Writes data/eval/retrieval_results.csv
Run:   python eval_retrieval.py
"""
import json

import numpy as np
import pandas as pd

import search as S

KS = (1, 5, 10)
MODES = ["dense", "bm25", "hybrid"]


def load_questions():
    rows = [json.loads(line) for line in open("data/eval/questions.jsonl", encoding="utf-8")
            if line.strip()]
    return pd.DataFrame([r for r in rows if not r["skip"]]).reset_index(drop=True)


def run(qs, mode, filtered):
    recs = []
    for _, q in qs.iterrows():
        kw = {}
        if filtered:
            month = q["filingDate"][:7]
            kw = dict(ticker=q["ticker"], form=q["form"], since=f"{month}-01", until=f"{month}-31")
        res = S.search(q["question"], k=10, mode=mode, **kw)
        ids = res["id"].tolist()
        rank = ids.index(q["chunk_id"]) + 1 if q["chunk_id"] in ids else np.nan
        top5 = res.head(5)
        doc5 = bool(((top5["ticker"] == q["ticker"]) & (top5["form"] == q["form"])
                     & (top5["section"] == q["section"])
                     & (top5["filingDate"] == q["filingDate"])).any())
        recs.append({"section": q["section"], "rank": rank, "doc5": doc5})
    return pd.DataFrame(recs)


def summarize(r):
    out = {f"hit@{k}": float(((r["rank"] <= k) & r["rank"].notna()).mean()) for k in KS}
    out["MRR"] = float((1 / r["rank"]).fillna(0).mean())
    out["doc_hit@5"] = float(r["doc5"].mean())
    return out


def main():
    qs = load_questions()
    print(f"{len(qs)} questions ({qs['section'].value_counts().to_dict()})\n")

    rows, per_section = [], []
    for filtered in (False, True):
        setting = "filtered" if filtered else "unfiltered"
        for mode in MODES:
            r = run(qs, mode, filtered)
            rows.append({"setting": setting, "mode": mode, **summarize(r)})
            if not filtered:
                ps = r.groupby("section").apply(
                    lambda g: float(((g["rank"] <= 5) & g["rank"].notna()).mean()),
                    include_groups=False)
                per_section.append(ps.rename(mode))

    res = pd.DataFrame(rows)
    res.to_csv("data/eval/retrieval_results.csv", index=False)
    for setting in ("unfiltered", "filtered"):
        print(f"--- {setting.upper()} ---")
        print(res[res["setting"] == setting].drop(columns="setting")
              .set_index("mode").round(3).to_string(), "\n")
    print("--- hit@5 by section, UNFILTERED ---")
    print(pd.concat(per_section, axis=1).round(3).to_string())


if __name__ == "__main__":
    main()
