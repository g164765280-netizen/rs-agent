# -*- coding: utf-8 -*-
"""通过 hf-mirror 下载 RAG 所需的 embedding 与 reranker 模型。"""
import os
import sys

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

from huggingface_hub import snapshot_download  # noqa: E402

MODELS = {
    "BAAI/bge-m3": "/media/sdb1/gzj/weights/llm/rscd/bge-m3",
    "BAAI/bge-reranker-v2-m3": "/media/sdb1/gzj/weights/llm/rscd/bge-reranker-v2-m3",
}

for repo, dest in MODELS.items():
    print(f"=== downloading {repo} -> {dest} ===", flush=True)
    try:
        snapshot_download(repo_id=repo, local_dir=dest)
        print(f"=== done {repo} ===", flush=True)
    except Exception as ex:
        print(f"=== FAILED {repo}: {ex} ===", flush=True)
        sys.exit(1)
print("ALL_MODELS_DONE", flush=True)
