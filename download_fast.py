# -*- coding: utf-8 -*-
"""把模型重下到 /home（NVMe，快盘），绕过 99% 满的 /media/sdb1 慢盘。"""
import os
import sys

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import snapshot_download  # noqa: E402

MODELS = {
    "Qwen/Qwen2.5-7B-Instruct": "/home/mcislab/weights/Qwen2.5-7B-Instruct",
    "BAAI/bge-m3": "/home/mcislab/weights/bge-m3",
    "BAAI/bge-reranker-v2-m3": "/home/mcislab/weights/bge-reranker-v2-m3",
}

for repo, dest in MODELS.items():
    print(f"=== downloading {repo} -> {dest} ===", flush=True)
    try:
        snapshot_download(repo_id=repo, local_dir=dest)
        print(f"=== done {repo} ===", flush=True)
    except Exception as ex:
        print(f"=== FAILED {repo}: {ex} ===", flush=True)
        sys.exit(1)
print("ALL_FAST_MODELS_DONE", flush=True)
