#!/usr/bin/env bash
set -Eeuo pipefail
cd /media/sdb1/gzj/support/llm/rscd
PY=/home/mcislab/anaconda3/envs/all-in-rag/bin/python
CORPUS=/media/sdb1/gzj/data/rscd/corpus
EMB=/media/sdb1/gzj/weights/llm/rscd/bge-m3
RERANK=/media/sdb1/gzj/weights/llm/rscd/bge-reranker-v2-m3
QBASE=/media/sdb1/gzj/weights/llm/MedicalGPT/Qwen2.5-7B-Instruct
SFT2=/media/sdb1/gzj/weights/llm/rscd/qwen25_7b_rscd_sft_lora_v2
SFT_MERGE=/media/sdb1/gzj/weights/llm/rscd/qwen25_7b_rscd_sft_merge
DPO1=/media/sdb1/gzj/weights/llm/rscd/qwen25_7b_rscd_dpo
OUT=/media/sdb1/gzj/generated_data/llm/rscd

# 1) 等 GPU 空闲
while [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sed '/^[[:space:]]*$/d' | wc -l)" -gt 0 ]; do
  echo "waiting for GPU ..."; sleep 60
done

# 2) 重跑基线（fixed eval）
echo "=== baseline re-eval (fixed eval) ==="
$PY -u eval_run.py --corpus "$CORPUS" --embedding "$EMB" --reranker "$RERANK" \
  --model "$QBASE" --configs oracle,rag --out "$OUT/results_baseline_v2eval.json" 2>&1 | tail -30

# 3) SFT v2 训练
echo "=== SFT v2 train (200 steps) ==="
bash run_rscd_sft_v2.sh 2>&1 | tail -15

# 4) SFT v2 重测
echo "=== SFT v2 re-eval ==="
$PY -u eval_run.py --corpus "$CORPUS" --embedding "$EMB" --reranker "$RERANK" \
  --model "$QBASE" --adapter "$SFT2" --configs oracle,rag --out "$OUT/results_sft_v2.json" 2>&1 | tail -30

# 5) DPO 重测（fixed eval）
echo "=== DPO re-eval (fixed eval) ==="
$PY -u eval_run.py --corpus "$CORPUS" --embedding "$EMB" --reranker "$RERANK" \
  --model "$SFT_MERGE" --adapter "$DPO1" --configs oracle,rag --out "$OUT/results_dpo_v2eval.json" 2>&1 | tail -30

echo "OVERNIGHT_DONE"
