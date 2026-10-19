#!/usr/bin/env bash
set -Eeuo pipefail
cd /media/sdb1/gzj/support/llm/rscd
PY=/home/mcislab/anaconda3/envs/all-in-rag/bin/python
CORPUS=/media/sdb1/gzj/data/rscd/corpus
EMB=/media/sdb1/gzj/weights/llm/rscd/bge-m3
RERANK=/media/sdb1/gzj/weights/llm/rscd/bge-reranker-v2-m3
QBASE=/media/sdb1/gzj/weights/llm/MedicalGPT/Qwen2.5-7B-Instruct
SFT_ADAPTER=/media/sdb1/gzj/weights/llm/rscd/qwen25_7b_rscd_sft_lora
SFT_MERGE=/media/sdb1/gzj/weights/llm/rscd/qwen25_7b_rscd_sft_merge
DPO_ADAPTER=/media/sdb1/gzj/weights/llm/rscd/qwen25_7b_rscd_dpo
OUT=/media/sdb1/gzj/generated_data/llm/rscd

echo "=== SFT re-eval ==="
$PY -u eval_run.py --corpus "$CORPUS" --embedding "$EMB" --reranker "$RERANK" \
  --model "$QBASE" --adapter "$SFT_ADAPTER" --configs oracle,rag \
  --out "$OUT/results_sft.json" 2>&1 | tail -40

echo "=== DPO re-eval ==="
$PY -u eval_run.py --corpus "$CORPUS" --embedding "$EMB" --reranker "$RERANK" \
  --model "$SFT_MERGE" --adapter "$DPO_ADAPTER" --configs oracle,rag \
  --out "$OUT/results_dpo.json" 2>&1 | tail -40

echo "ALL_REEVAL_DONE"
