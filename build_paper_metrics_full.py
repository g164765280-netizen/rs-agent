# -*- coding: utf-8 -*-
"""权威指标扩充：识别指标库全部论文的「本文方法」，生成完整 paper_metrics。

- 现状：paper_metrics.json 只有 30 篇（eval_set_v9 截断）
- 本脚本：识别指标库覆盖的 ~60 篇的本文方法，生成完整 paper_metrics_full.json
- 后续：补抽剩余 39 篇的表，再扩到 109 篇

用法：
  /media/sdb1/gzj/support/llm/envs/paperqa/bin/python build_paper_metrics_full.py
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import litellm

LIB = Path("/media/sdb1/gzj/data/rscd/corpus/metric_library.jsonl")
FULLTEXT = Path("/media/sdb1/gzj/data/rscd/corpus/fulltext")
OUT = Path("/media/sdb1/gzj/support/llm/rscd/paper_metrics_full.json")

API_KEY = os.environ.get("OPENAI_API_KEY", "")
API_BASE = os.environ.get("OPENAI_API_BASE", "")
MODEL = "openai/qwen3.8-max"

TITLE = {}
for ft in FULLTEXT.glob("*.json"):
    d = json.loads(ft.read_text(encoding="utf-8"))
    TITLE[d["arxiv_id"]] = d.get("title", "").strip()


def identify_proposed(batch: list[dict]) -> dict[str, str]:
    lines = []
    for b in batch:
        lines.append(f"[{b['paper_id']}] 标题: {b['title']}\n  方法名: {', '.join(b['methods'][:25])}")
    prompt = ("下面是若干篇遥感变化检测论文。每篇列出标题和它实验结果表里出现的方法名。\n"
              "对每篇论文，指出哪【一个】方法名是「本文提出的方法」（不是基线、不是对比方法）。\n"
              "判断依据：方法名含 Ours/Proposed/our，或与论文标题里的方法缩写一致。\n\n"
              + "\n\n".join(lines) +
              "\n\n只输出 JSON：{\"paper_id\": \"本文方法名\", ...}，无法判断的 paper_id 不要写进 JSON。")
    r = litellm.completion(model=MODEL, api_key=API_KEY, api_base=API_BASE,
                           messages=[{"role": "user", "content": prompt}],
                           temperature=0.0, max_tokens=1500, extra_body={"enable_thinking": False})
    raw = r.choices[0].message.content or ""
    try:
        return json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
    except Exception:
        return {}


def main():
    by_paper: dict[str, dict] = {}
    for l in LIB.read_text(encoding="utf-8").splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        p = by_paper.setdefault(r["paper_id"], {"methods": set(), "recs": []})
        p["methods"].add(r["method"])
        p["recs"].append(r)

    papers = [(pid, TITLE.get(pid, ""), sorted(v["methods"])) for pid, v in by_paper.items()]
    papers = [p for p in papers if p[1]]
    print(f"指标库覆盖 {len(papers)} 篇论文，开始识别本文方法...", flush=True)

    proposed = {}
    t0 = time.time()
    for i in range(0, len(papers), 5):
        batch = [{"paper_id": p[0], "title": p[1], "methods": p[2]} for p in papers[i:i+5]]
        try:
            res = identify_proposed(batch)
            proposed.update(res)
        except Exception as e:
            print(f"  批 {i//5+1} 失败: {str(e)[:80]}", flush=True)
        print(f"  进度 {min(i+5, len(papers))}/{len(papers)}，已识别 {len(proposed)} 篇（{time.time()-t0:.0f}s）", flush=True)
        time.sleep(1)

    # 生成完整 paper_metrics：每篇取本文方法的所有 f1/iou/oa 记录（多数据集）
    out = {}
    for pid, v in by_paper.items():
        m = proposed.get(pid)
        if not m:
            continue
        recs = []
        for r in v["recs"]:
            if r["method"] == m and r["value"] not in ("-", "", "nan", None):
                recs.append({
                    "method": m, "dataset": r.get("dataset"),
                    "metric": r.get("metric"), "value": r.get("value"),
                    "title": TITLE.get(pid, ""),
                })
        if recs:
            out[pid] = recs

    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    n_papers = len(out)
    n_recs = sum(len(v) for v in out.values())
    print(f"\n生成 paper_metrics_full.json：{n_papers} 篇论文，{n_recs} 条本文方法指标", flush=True)


if __name__ == "__main__":
    main()