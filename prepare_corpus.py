# -*- coding: utf-8 -*-
"""切分留出/语料 + 解析 PDF + 分块 + 建 BM25/FAISS 索引。

用法（all-in-rag 环境）：
  python prepare_corpus.py \
    --corpus /media/sdb1/gzj/data/rscd/corpus \
    --embedding /media/sdb1/gzj/weights/llm/rscd/bge-m3 \
    --heldout-ratio 0.40 --seed 42
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
from pathlib import Path

import numpy as np


def tokenize(text: str):
    """中英文混合分词：英文按词，中文用 jieba（若可用），否则按字。"""
    try:
        import jieba  # type: ignore
        toks = [t for t in jieba.cut(text.lower()) if t.strip()]
    except Exception:
        toks = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text.lower())
    return toks


def read_metadata(corpus: Path) -> list[dict]:
    papers: dict[str, dict] = {}
    for mf in sorted(corpus.glob("metadata*.jsonl")):
        for line in mf.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            aid = rec.get("arxiv_id", "")
            if aid and aid not in papers:
                papers[aid] = rec
    return list(papers.values())


def extract_chunks(papers: list[dict], corpus: Path, chunk_size: int = 520) -> list[dict]:
    import fitz  # pymupdf
    chunks = []
    for p in papers:
        aid = p["arxiv_id"].replace("/", "_")
        pdf = corpus / "pdf" / f"{aid}.pdf"
        if not pdf.exists():
            continue
        try:
            doc = fitz.open(pdf)
        except Exception:
            continue
        for pno in range(len(doc)):
            try:
                text = doc[pno].get_text()
            except Exception:
                continue
            paras = [x.strip() for x in re.split(r"\n\s*\n", text) if x.strip()]
            buf = ""
            for para in paras:
                if len(buf) + len(para) + 1 <= chunk_size:
                    buf = (buf + "\n" + para).strip()
                else:
                    if buf:
                        chunks.append({
                            "paper_id": p["arxiv_id"],
                            "title": p.get("title", ""),
                            "page": pno,
                            "text": buf,
                        })
                    buf = para
            if buf.strip():
                chunks.append({
                    "paper_id": p["arxiv_id"],
                    "title": p.get("title", ""),
                    "page": pno,
                    "text": buf,
                })
        doc.close()
    return chunks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--embedding", required=True)
    ap.add_argument("--heldout-ratio", type=float, default=0.40)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    corpus = Path(args.corpus)
    index_dir = corpus / "index"
    index_dir.mkdir(parents=True, exist_ok=True)

    papers = read_metadata(corpus)
    print(f"total papers (metadata): {len(papers)}", flush=True)

    # 确定性切分：按 arxiv_id 哈希，隔离到整篇论文粒度
    heldout, train = [], []
    for p in papers:
        h = int(hashlib.sha256(p["arxiv_id"].encode()).hexdigest(), 16) % 1000
        (heldout if h < int(args.heldout_ratio * 1000) else train).append(p)
    held_ids = {p["arxiv_id"] for p in heldout}
    train_ids = {p["arxiv_id"] for p in train}
    print(f"heldout papers: {len(heldout)}, train papers: {len(train)}", flush=True)

    split = {
        "heldout_ratio": args.heldout_ratio,
        "heldout_ids": sorted(held_ids),
        "train_ids": sorted(train_ids),
        "heldout_meta": heldout,
        "train_meta": train,
    }
    (corpus / "split.json").write_text(json.dumps(split, ensure_ascii=False, indent=2), encoding="utf-8")

    # 解析 + 分块（仅训练集建索引）
    print("extracting chunks from train papers ...", flush=True)
    chunks = extract_chunks(train, corpus)
    print(f"total chunks: {len(chunks)}", flush=True)
    if not chunks:
        print("ERROR: no chunks", flush=True)
        return

    (index_dir / "chunks.jsonl").write_text(
        "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in chunks), encoding="utf-8")

    # BM25
    from rank_bm25 import BM25Okapi
    tokenized = [tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    with open(index_dir / "bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)

    # embeddings + FAISS
    from sentence_transformers import SentenceTransformer
    print("loading embedding model ...", flush=True)
    model = SentenceTransformer(args.embedding)
    texts = [c["text"] for c in chunks]
    print("embedding chunks ...", flush=True)
    embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False,
                        batch_size=32, convert_to_numpy=True)
    np.save(index_dir / "embeddings.npy", embs.astype("float32"))

    import faiss
    d = embs.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embs.astype("float32"))
    faiss.write_index(index, str(index_dir / "faiss.index"))

    print(f"DONE: {len(chunks)} chunks, {len(train)} train papers indexed", flush=True)


if __name__ == "__main__":
    main()
