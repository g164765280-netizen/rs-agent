# -*- coding: utf-8 -*-
"""快速测试 bge-reranker-v2-m3 能否在 GPU1 上正常 predict（不加载 Qwen）。"""
import os
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from sentence_transformers import CrossEncoder

pairs = [("变化检测常用哪些评价指标", "该论文在 LEVIR-CD 上报告 F1=0.892、IoU=0.805，采用 Precision、Recall、F1 与 IoU 作为评价指标。"),
         ("变化检测常用哪些评价指标", "本文提出一种基于注意力的建筑变化检测方法。")]

print("loading reranker ...", flush=True)
t0 = time.time()
m = CrossEncoder("/media/sdb1/gzj/weights/llm/rscd/bge-reranker-v2-m3", device="cuda:1")
print(f"loaded in {time.time()-t0:.1f}s", flush=True)
t0 = time.time()
scores = m.predict(pairs)
print(f"predict in {time.time()-t0:.1f}s, scores={scores}", flush=True)
print("OK", flush=True)
