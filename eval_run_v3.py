# -*- coding: utf-8 -*-
"""评测 v3：无泄漏检索 + 分离指标（修复 v2 的答案泄漏）。

- rag 模式：检索只看 question 文本，绝不访问 gold（gold 只用于打分）。
- oracle 模式：给对证据，测生成上限。
- 分离三个指标：
  - Document Recall@k：正确论文是否被召回
  - Evidence Recall@k：含答案数字的句子是否被召回
  - Answer Accuracy：最终答案的数字是否正确（绑定到正确的数据集）
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from rag_pipeline import RAGPipeline


def nums_in(text: str) -> list[float]:
    return [float(x) for x in re.findall(r"\d+\.\d+|\d+", text)]


def num_match(gold_str: str, answer_nums: list[float]) -> bool:
    g = float(gold_str)
    for a in answer_nums:
        if abs(a - g) <= max(0.001, g * 0.02):      # 同尺度（0.863 vs 0.863/0.86）
            return True
        if abs(a - g * 100) <= 0.5:                 # 答案用百分数（86.3 vs 0.863）
            return True
        if abs(g - a * 100) <= 0.005:               # gold 百分数，答案小数
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--embedding", required=True)
    ap.add_argument("--reranker", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--configs", default="rag")
    ap.add_argument("--out", default="")
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    corpus = Path(args.corpus)
    qs = [json.loads(l) for l in (corpus / "eval_set_v3.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        qs = qs[: args.limit]
    print(f"number questions: {len(qs)}", flush=True)

    pipe = RAGPipeline(corpus, args.embedding, args.reranker, args.model)

    results = {}
    for cfg in args.configs.split(","):
        cfg = cfg.strip()
        doc_rec = ev_rec = ans_acc = 0
        t0 = time.time()
        for i, q in enumerate(qs, 1):
            gold = q["gold"]
            if cfg == "oracle":
                ctx = [{"paper_id": q["paper_id"], "title": q["title"],
                        "page": 0, "text": gold["evidence"]}]
            else:
                # 无泄漏：检索只看问题文本
                ctx = pipe.retrieve(q["question"], top_k=args.top_k * 2, rerank_top=args.top_k)

            # 打分（gold 只在这里用，不喂给模型）
            d_rec = q["paper_id"] in {c["paper_id"] for c in ctx}
            e_rec = any(gold["value"] in c["text"] for c in ctx)

            ans = pipe.generate(q["question"], ctx)
            a_acc = num_match(gold["value"], nums_in(ans))

            doc_rec += d_rec
            ev_rec += e_rec
            ans_acc += a_acc
            if i % 10 == 0 or i == len(qs):
                print(f"  [{cfg}][{i}/{len(qs)}] DocRec={doc_rec/i:.2f} "
                      f"EvRec={ev_rec/i:.2f} AnsAcc={ans_acc/i:.2f} ({time.time()-t0:.0f}s)", flush=True)

        n = len(qs)
        result = {"config": cfg, "n": n, "top_k": args.top_k,
                  "doc_recall": doc_rec / n, "ev_recall": ev_rec / n,
                  "answer_acc": ans_acc / n}
        results[cfg] = result
        print(f"=== {cfg} ===", flush=True)
        print(f"  Document Recall@{args.top_k}: {doc_rec}/{n} = {doc_rec/n:.1%}", flush=True)
        print(f"  Evidence Recall@{args.top_k}: {ev_rec}/{n} = {ev_rec/n:.1%}", flush=True)
        print(f"  Answer Accuracy: {ans_acc}/{n} = {ans_acc/n:.1%}", flush=True)

    if args.out:
        Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"results written to {args.out}", flush=True)


if __name__ == "__main__":
    main()
