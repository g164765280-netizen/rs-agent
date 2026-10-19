# -*- coding: utf-8 -*-
"""多专家辩论人格（Panel v2）：4 视角专家 + 主持人，多轮辩论。

相比单轮会诊，新增「交叉质疑」轮（借鉴 Tree-of-Debate 的辩论核心）：
  轮1 发言：4 视角各自独立分析
  轮2 质疑：每个专家针对其他 3 位的观点提出质疑/反驳/补充
  轮3 总结：主持人综合「发言 + 质疑」，给出最终结论

4 视角：
  - 方法派：只看方法设计思路
  - 指标派：只看实验结果数据
  - 综述派：看领域演进脉络
  - 质疑派：挑刺、找反例

可插拔：LLM 走 LLMBackend（本地 8B / API 切换）。
"""
from __future__ import annotations

from persona_base import PersonaReply
from llm_backend import get_backend

PANEL_VIEWS = [
    ("方法派", "你从【方法设计思路】角度分析：各自的核心机制、创新点、实现复杂度。"),
    ("指标派", "你从【实验结果数据】角度分析：各自在数据集上的指标表现、谁更好、差多少。"),
    ("综述派", "你从【领域演进脉络】角度分析：这两个方法在变化检测发展史里的位置、承上启下关系。"),
    ("质疑派", "你从【挑刺】角度分析：各自的局限、被忽略的问题、可能的反例或翻车场景。"),
]

CRITIQUE_PROMPT = """你是遥感变化检测领域的{name}专家。下面是【其他 3 位专家】对同一问题的观点，请针对他们的观点提出你的【质疑/反驳/补充】：

【问题】{question}

【其他专家观点】
{others}

请针对他们观点中的：①可能的错误或片面之处 ②被你视角发现的遗漏 ③需要补充的关键点，给出你的质疑（2-3 句，直接说，不要客套）。"""

MODERATOR_PROMPT = """你是会诊主持人。下面 4 位专家先各自发言，然后又互相质疑了一轮。请综合【发言 + 质疑】两个阶段，给出最终结论。

【问题】{question}

【阶段一：各自发言】
{views}

【阶段二：交叉质疑】
{critiques}

请输出：
1. 【结论】一句话回答用户的问题（哪个更好/核心区别是什么）
2. 【关键分歧】专家之间最主要的争议点（谁质疑了谁）
3. 【建议】如果用户要选，该怎么选（给出判断依据）"""


class PanelPersona:
    name = "panel"
    system_prompt = "你是多专家辩论主持人，综合方法派/指标派/综述派/质疑派的发言与质疑给结论。"

    def __init__(self, backend=None):
        self.llm = backend or get_backend("local")

    def can_activate(self, question: str) -> float:
        kw = ["哪个好", "哪个更好", "谁更强", "谁更好", "优劣", "哪个更优",
              "哪个强", "优缺点", "vs", "versus", "更强", "更好用", "哪个更",
              "还是", "谁好"]
        if any(k in question.lower() for k in kw):
            return 0.85
        return 0.2

    def respond(self, history: list[dict] | None, question: str) -> PersonaReply:
        # 轮1：4 视角各自发言
        views = []
        for view_name, view_instr in PANEL_VIEWS:
            prompt = (
                f"你是遥感变化检测领域的{view_name}专家。\n{view_instr}\n\n"
                f"问题：{question}\n\n"
                f"请用 2-4 句话给出你的分析，直接说观点，不要客套。"
            )
            text = self.llm.generate(
                [{"role": "user", "content": prompt}], max_tokens=300, temperature=0.3
            )
            views.append(f"【{view_name}】{text}")

        # 轮2：交叉质疑（每个专家针对其他 3 位观点质疑）
        critiques = []
        for i, (view_name, _) in enumerate(PANEL_VIEWS):
            others = "\n\n".join(v for j, v in enumerate(views) if j != i)
            prompt = CRITIQUE_PROMPT.format(name=view_name, question=question, others=others)
            text = self.llm.generate(
                [{"role": "user", "content": prompt}], max_tokens=300, temperature=0.3
            )
            critiques.append(f"【{view_name}质疑】{text}")

        # 轮3：主持人综合发言+质疑
        views_text = "\n\n".join(views)
        critiques_text = "\n\n".join(critiques)
        final = self.llm.generate(
            [{"role": "system", "content": self.system_prompt},
             {"role": "user", "content": MODERATOR_PROMPT.format(
                 question=question, views=views_text, critiques=critiques_text)}],
            max_tokens=800, temperature=0.2
        )
        # 拼接：主持人结论 + 发言原文 + 质疑原文
        full = (final + "\n\n---\n\n### 阶段一：各专家发言\n\n" + views_text
                + "\n\n---\n\n### 阶段二：交叉质疑\n\n" + critiques_text)
        return PersonaReply(text=full, persona=self.name, tool_used=None,
                            meta={"n_views": len(views), "n_critiques": len(critiques),
                                  "rounds": 2})