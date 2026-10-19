# -*- coding: utf-8 -*-
"""自定义 context_serializer：关摘要后用 cross-encoder reranker 重排过滤。

v6 用 cosine 相似度（bge-m3）重排，结果 grounded 只到 16%——因为 cosine 相似度
「语义相关 ≠ 内容对口」。本模块改用 cross-encoder（bge-reranker-v2-m3），
把「问题 + chunk」拼接后联合打分，能更精确判断 chunk 是否真的能回答问题。

签名需匹配 paperqa 的 custom_context_serializer：
  async def(settings, contexts, question, pre_str) -> str
"""
from __future__ import annotations

_reranker = None


def _get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder("/media/sdb1/gzj/weights/llm/rscd/bge-reranker-v2-m3")
    return _reranker


def _score_contexts(contexts, question: str) -> list[tuple[float, object]]:
    """用 cross-encoder 给每个 (question, chunk) 联合打分。"""
    model = _get_reranker()
    texts = [c.text.text for c in contexts]
    pairs = [(question, t) for t in texts]
    scores = model.predict(pairs)
    return list(zip(scores.tolist(), contexts))


async def embedding_rerank_serializer(settings, contexts, question: str, pre_str):
    """自定义 serializer：cross-encoder 重排 + 过滤，复用 paperqa 拼接格式。"""
    answer_config = settings.answer
    prompt_config = settings.prompts

    scored = _score_contexts(list(contexts), question)
    scored.sort(key=lambda x: -x[0])
    top = scored[: answer_config.answer_max_sources]
    # cross-encoder 分数一般越高越相关；取正分（>0）视为相关
    filtered = [(sim, c) for sim, c in top if sim > 0]

    if not filtered:
        filtered = scored[:1] if scored else []

    context_inner_prompt = prompt_config.context_inner
    inner_strs = [
        context_inner_prompt.format(
            name=c.id,
            text=c.context,
            citation=c.text.doc.formatted_citation,
            **(c.model_extra or {}),
        )
        for sim, c in filtered
    ]
    context_str_body = "\n\n".join(inner_strs)

    if pre_str:
        context_str_body += f"\n\nExtra background information: {pre_str}"

    return prompt_config.context_outer.format(
        context_str=context_str_body,
        valid_keys=", ".join([c.id for sim, c in filtered]),
    )