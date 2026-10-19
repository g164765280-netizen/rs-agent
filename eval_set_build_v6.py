# -*- coding: utf-8 -*-
"""出题 v6：纯正则（不编造），每题答案唯一。

- 正则抽「指标→数值→一句话证据」，过滤垃圾值。
- 同一论文「同一数据集+同一指标」若出现多个数值 → 不唯一 → 跳过。
- 题目：论文《X》在 {dataset} 上的 {metric} 是多少？
- gold：{dataset, metric, value, evidence}，value 必在 evidence 原文里。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

CORPUS = Path("/media/sdb1/gzj/data/rscd/corpus")

METRIC_RE = re.compile(
    r"(F1|IoU|mIoU|OA|Precision|Recall)(?:-?score)?\s*(?:of)?\s*[=:]?\s*(\d+(?:\.\d+)?)\s*%?",
    re.IGNORECASE)
KNOWN_DATASETS = ["LEVIR-CD", "WHU-CD", "S2Looking", "SYSU-CD", "CDD", "DSIFN", "SECOND",
                  "HRSCD", "CLCD", "OSCD", "LEVIR", "WHU", "xView2"]


def valid_value(num_str: str) -> bool:
    if "." in num_str:
        return True
    n = float(num_str)
    return 10 <= n <= 100


def extract_entries(text: str) -> list[dict]:
    out, seen = [], set()
    sentences = re.split(r"(?<=[.!?])\s+|\n", text)
    for m in METRIC_RE.finditer(text):
        field, num = m.group(1).lower(), m.group(2)
        if (field, num) in seen or not valid_value(num):
            continue
        seen.add((field, num))
        # 干净的一句（含数字）
        ev = text[m.start():m.end()]
        for s in sentences:
            if m.group(0) in s or (field in s and num in s):
                ev = s.strip()
                break
        # 用 ±250 窗口找数据集
        w = text[max(0, m.start() - 250):min(len(text), m.end() + 250)]
        ds = [d for d in KNOWN_DATASETS if re.search(re.escape(d), w, re.IGNORECASE)]
        if ds:
            out.append({"metric": field, "value": num, "dataset": ds[0], "sentence": ev})
    return out


def main():
    qs = []
    skipped_ambiguous = 0
    for ft in sorted((CORPUS / "fulltext").glob("*.json")):
        d = json.loads(ft.read_text(encoding="utf-8"))
        title = d.get("title", "").strip()
        entries = extract_entries("\n".join(d.get("pages_text", [])))
        # 按 (dataset, metric) 分组，只保留"唯一数值"的
        groups = {}
        for e in entries:
            groups.setdefault((e["dataset"], e["metric"]), []).append(e)
        for (ds, mt), es in groups.items():
            vals = set(e["value"] for e in es)
            if len(vals) != 1:
                skipped_ambiguous += 1
                continue
            e = es[0]
            qs.append({
                "paper_id": d["arxiv_id"], "title": title,
                "question": f"论文《{title}》在 {ds} 上的 {mt} 是多少？",
                "gold": {"dataset": ds, "metric": mt, "value": e["value"], "evidence": e["sentence"]},
            })
    out = CORPUS / "eval_set_v6.jsonl"
    out.write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in qs), encoding="utf-8")
    print(f"共 {len(qs)} 道唯一答案数字题（跳过歧义 {skipped_ambiguous} 组），存至 {out}")


if __name__ == "__main__":
    main()