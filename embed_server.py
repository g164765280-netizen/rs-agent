# -*- coding: utf-8 -*-
"""OpenAI 兼容的 bge-m3 embedding 服务（FastAPI + sentence-transformers）。

供 litkg-assistant 的 vector_store 调用（client.embeddings.create）。
"""
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI()
model = SentenceTransformer("/media/sdb1/gzj/weights/llm/rscd/bge-m3", device="cuda:1")


class Req(BaseModel):
    input: list | str
    model: str = "bge-m3"


@app.post("/v1/embeddings")
def embed(req: Req):
    texts = [req.input] if isinstance(req.input, str) else req.input
    vecs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    return {
        "object": "list",
        "data": [{"object": "embedding", "embedding": v.tolist(), "index": i}
                 for i, v in enumerate(vecs)],
        "model": req.model,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
