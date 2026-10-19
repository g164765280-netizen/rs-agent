# -*- coding: utf-8 -*-
"""出题 v5：用 LLM 从论文抽"论文自己提出的方法"的实验结果，保证每题答案唯一。

- 题目：论文《X》提出的方法在 {dataset} 上的 {metric} 是多少？
- gold：{dataset, metric, value, evidence}（唯一答案）
- 只抽"自己方法"的结果，不抽基线，抽不明确就跳过。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CORPUS = Path("/media/sdb1/gzj/data/rscd/corpus")
MODEL_PATH = "/media/sdb1/yjl/ollama-models/Qwen3-8B"

# 全局加载一次
_tok = None
_model = None


def _get_model():
    global _tok, _model
    if _model is None:
        _tok = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True, trust_remote_code=False)
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, local_files_only=True, trust_remote_code=False,
            torch_dtype=torch.bfloat16, device_map="cuda:0")
        _model.eval()
    return _tok, _model

EXTRACT_PROMPT = """你是遥感变化检测论文的分析员。从论文文本里，提取**论文自己提出的那个方法**的关键实验结果（只提取"自己方法/本文方法/Ours/our method"的结果，不要基线方法、不要对比方法的结果）。

只输出 JSON，不要解释，格式：
{"results": [{"dataset": "数据集标准名", "metric": "f1/iou/oa/precision/recall/miou", "value": "数值", "evidence": "包含该数值的原文句子"}]}

规则：
1. 只提取"本文提出的方法 / Ours / the proposed method"的结果，基线、SOTA 对比、消融的其它变体都不提取。
2. dataset 用标准名：LEVIR-CD、WHU-CD、S2Looking、SYSU-CD、CDD、DSIFN、SECOND、HRSCD、CLCD、OSCD。
3. metric 全小写：f1、iou、oa、precision、recall、miou。
4. value 照抄原文数字。
5. evidence 照抄"包含该数值的那一句原文"。
6. 只有"明确知道是本文方法在某个数据集上的结果"才提取；不确定就不提取。"""


def extract(text: str) -> list[dict]:
    tok, model = _get_model()
    msgs = [{"role": "system", "content": "你是专业的论文实验数据抽取器。只输出 JSON，不要思考、不要解释。"},
            {"role": "user", "content": EXTRACT_PROMPT + "\n\n论文文本：\n" + text[:10000]}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(ids, max_new_tokens=1500, do_sample=False)
    raw = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    try:
        d = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
        return d.get("results", [])
    except Exception:
        return []


def main(limit: int = 5):
    qs = []
    fts = sorted((CORPUS / "fulltext").glob("*.json"))[:limit]
    t0 = time.time()
    for ft in fts:
        d = json.loads(ft.read_text(encoding="utf-8"))
        title = d.get("title", "").strip()
        text = "\n".join(d.get("pages_text", []))
        results = extract(text)
        for r in results:
            qs.append({
                "paper_id": d["arxiv_id"], "title": title,
                "question": f"论文《{title}》提出的方法在 {r['dataset']} 上的 {r['metric']} 是多少？",
                "gold": {"dataset": r["dataset"], "metric": r["metric"],
                         "value": r["value"], "evidence": r["evidence"]},
            })
        print(f"[{fts.index(ft)+1}/{limit}] {title[:40]} → {len(results)} 条（{time.time()-t0:.0f}s）", flush=True)
        for r in results[:4]:
            print(f"    - {r['dataset']} {r['metric']}={r['value']}", flush=True)

    out = CORPUS / "eval_set_v5.jsonl"
    out.write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in qs), encoding="utf-8")
    print(f"\n共 {len(qs)} 道无歧义数字题，存至 {out}")


if __name__ == "__main__":
    import sys
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 5)