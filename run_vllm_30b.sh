#!/bin/bash
CUDA_VISIBLE_DEVICES=1 /home/mcislab/anaconda3/envs/vllm20/bin/python -m vllm.entrypoints.openai.api_server \
  --model /media/sdb1/yjl/ollama-models/Qwen3-30B-A3B-Instruct-2507-Int4-W4A16 \
  --served-model-name Qwen3-30B-A3B \
  --port 8001 \
  --max-model-len 2048 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 8 \
  --enforce-eager > /media/sdb1/gzj/support/llm/rscd/vllm_30b.log 2>&1
