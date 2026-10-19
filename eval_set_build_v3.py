# -*- coding: utf-8 -*-
"""评测集 v3：无歧义数字题（含数据集/方法上下文）+ 无泄漏数据结构。

修复 v2 的两个问题：
1. 数字题歧义：一篇论文多个 F1，但题目文本相同、答案不同
   → 题目必须含「在 {dataset} 上」，gold 记录 metric/value/dataset/method/evidence
2. 供评测端无泄漏使用：本脚本只生成数据；检索端只读 question 文本，不碰 gold。
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
KNOWN_METHODS = ["ChangeFormer", "BIT", "SNUNet", "TinyCD", "Changer", "STANet", "DSIFN",
                 "IFNet", "USSFC-Net", "HANet", "DMINet", "SiamConc", "SiamDiff", "A2Net",
                 "FC-Siam", "FC-EF", "FC-Siam-diff", "LightCDNet", "SSCDNet"]


def method_from_title(title: str) -> str:
    m = re.search(r"([A-Za-z][A-Za-z0-9\-]+Net|[A-Za-z][A-Za-z0-9\-]+Former|[A-Za-z][A-Za-z0-9]+Net)", title)
    return m.group(1) if m else ""


def extract_numbers(text: str, title: str) -> list[dict]:
    """提取 (metric, value, dataset, method, evidence)，取数字周围 ±250 字符窗口找数据集消歧。"""
    out, seen = [], set()
    title_method = method_from_title(title)
    for m in METRIC_RE.finditer(text):
        field, num = m.group(1).lower(), m.group(2)
        if (field, num) in seen:
            continue
        seen.add((field, num))
        s = max(0, m.start() - 250)
        e = min(len(text), m.end() + 250)
        ctx = text[s:e].replace("\n", " ").strip()
        ds = [d for d in KNOWN_DATASETS if re.search(re.escape(d), ctx, re.IGNORECASE)]
        if not ds:
            continue
        meth = ""
        for mn in KNOWN_METHODS:
            if re.search(re.escape(mn), ctx, re.IGNORECASE):
                meth = mn
                break
        if not meth and title_method and re.search(re.escape(title_method), ctx, re.IGNORECASE):
            meth = title_method
        out.append({"metric": field, "value": num, "dataset": ds[0],
                    "method": meth, "evidence": ctx})
    return out


def main():
    split = json.loads((CORPUS / "split.json").read_text(encoding="utf-8"))
    held_ids = set(split["heldout_ids"])
    qs = []
    for ft in sorted((CORPUS / "fulltext").glob("*.json")):
        d = json.loads(ft.read_text(encoding="utf-8"))
        sp = "heldout" if d["arxiv_id"] in held_ids else "train"
        title = d.get("title", "").strip()
        full_text = "\n".join(d.get("pages_text", []))
        if not title or not full_text:
            continue
        for n in extract_numbers(full_text, title)[:5]:
            qs.append({
                "type": "number",
                "paper_id": d["arxiv_id"],
                "title": title,
                "split": sp,
                "question": f"论文《{title}》在 {n['dataset']} 上报告的 {n['metric']} 是多少？",
                "gold": {"metric": n["metric"], "value": n["value"],
                         "dataset": n["dataset"], "method": n["method"],
                         "evidence": n["evidence"]},
            })
    out = CORPUS / "eval_set_v3.jsonl"
    out.write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in qs), encoding="utf-8")
    print(f"total number questions: {len(qs)}")
    with_method = sum(1 for q in qs if q["gold"]["method"])
    print(f"with method: {with_method}/{len(qs)}")
    print(f"datasets: {sorted(set(q['gold']['dataset'] for q in qs))}")


if __name__ == "__main__":
    main()
