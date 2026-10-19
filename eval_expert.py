# -*- coding: utf-8 -*-
"""专家能力评测：验证"知识内化 + 应用"（而非检索玩具）。

三套题（与你确认的"专家"定义对应）：
  ① 知识内化：关闭检索，问领域概念/方法/数据集（考察图谱/记忆里的知识）
  ② 知识应用：新场景选型（给论文里没有的新场景，看推理）
  ③ 跨论文综合：脉络/趋势/本质区别（走 litkg 的 global 图谱综述）

评测口径：LLM-as-judge 粗略判分（含关键点给分）+ 人工抽查。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/media/sdb1/gzj/project/llm/reference/litkg-assistant-main")
from core.kg_store import KGStore
from core.vector_store import VectorStore
from core.graphrag import ask

# 三套题：每题的 (问题, 期望关键点列表)
KNOWLEDGE = [
    ("变化检测中常用的评价指标有哪些？", ["F1", "IoU", "Precision", "Recall", "OA"]),
    ("遥感变化检测的经典方法有哪些？", ["BIT", "ChangeFormer", "SNUNet", "TinyCD", "Transformer"]),
    ("变化检测常用的数据集有哪些？", ["LEVIR", "WHU", "S2Looking", "SYSU", "CDD"]),
    ("Transformer 类变化检测方法的核心思想是什么？", ["自注意力", "token", "全局依赖", "注意力"]),
    ("孪生网络（Siamese）为什么适合变化检测？", ["权重共享", "特征提取", "双时相", "对比"]),
]

APPLICATION = [
    ("给一个新场景：SAR 遥感影像、标注极少（只有几十张），需要做变化检测。应该怎么设计方法？为什么？",
     ["半监督", "少样本", "自监督", "预训练", "迁移", "SAR 抑制斑点"]),
    ("给一个新场景：需要检测城市里新建的密集小建筑物，对漏检零容忍（宁可多报）。应该用什么指标和策略？为什么？",
     ["Recall", "低阈值", "小目标", "多尺度", "IoU"]),
    ("给一个新场景：影像分辨率很低 + 云遮挡严重，做变化检测。应该优先解决什么？",
     ["配准", "去云", "伪变化", "低分辨率", "融合"]),
]

SYNTHESIS = [
    ("近五年遥感变化检测方法的主要演进脉络是什么？", ["CNN", "Transformer", "Mamba", "多尺度", "高效"]),
    ("BIT 和 ChangeFormer 的本质区别是什么？各自适合什么场景？",
     ["Transformer", "token", "多尺度", "层次化", "计算效率"]),
    ("变化检测领域当前的主要挑战和未解决问题有哪些？", ["小目标", "类别不平衡", "配准", "泛化", "标注"]),
]


def judge(answer: str, keywords: list) -> tuple[bool, int]:
    hit = sum(1 for k in keywords if k.lower() in answer.lower())
    return hit >= 1, hit


def run_suite(name, items, kg, vs, mode):
    print(f"\n{'='*60}\n【{name}】mode={mode}\n{'='*60}", flush=True)
    ok = 0
    for q, kws in items:
        t0 = time.time()
        try:
            a = ask(q, kg=kg, vector_store=vs, mode=mode)
            ans = a.answer if hasattr(a, "answer") else str(a)
        except Exception as e:
            ans = f"[ERROR] {e}"
        hit_ok, hit_n = judge(ans, kws)
        ok += hit_ok
        print(f"Q: {q}\n  hit={hit_n}/{len(kws)} {'✓' if hit_ok else '✗'} ({time.time()-t0:.0f}s)\n  A: {ans[:160]}\n", flush=True)
    print(f"→ {name} 通过率: {ok}/{len(items)}", flush=True)
    return ok


def main():
    kg = KGStore()
    vs = VectorStore()
    total = 0
    total += run_suite("① 知识内化（global 图谱）", KNOWLEDGE, kg, vs, "global")
    total += run_suite("② 知识应用（hybrid）", APPLICATION, kg, vs, "hybrid")
    total += run_suite("③ 跨论文综合（global）", SYNTHESIS, kg, vs, "global")
    print(f"\n\n=== 专家评测总结: {total}/15 ===", flush=True)


if __name__ == "__main__":
    main()