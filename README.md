# RS-Agent · 遥感变化检测多智能体 RAG 系统

遥感变化检测（Remote Sensing Change Detection）领域的多智能体论文问答与变化分析系统（个人项目）。编排者 Qwen3-8B 调度 **6 个能力 Agent**（文本检索 / 数字查表 / 影像解读 / 变化检测 / 知识融合 / 报告生成），自建三代评测体系 + RAGAS/RAGChecker 双框架交叉验证，走完 SFT→merge→DPO→GRPO 对齐全链路。

> 项目目标不是做 demo，而是把整条链路走通、每一环拿到可信的数字：**每个结论有对照组，每个数字能对账**。

**技术栈**：Qwen3-8B · Qwen2.5-7B-Instruct · Qwen3-VL-8B · bge-m3 · bge-reranker-v2-m3 · BM25(jieba) · FAISS · LlamaIndex · PyMuPDF · Flask · FastMCP · vLLM · PEFT/TRL · RAGAS · RAGChecker

**在线展示页**：[rs-agent-showcase](https://github.com/g164765280-netizen/rs-agent-showcase)（React + Vite 项目展示，含评测可视化）

---

## 架构一张图

```
                    用户（问题 / 0-2 张遥感影像）
                              │
                ┌─────────────▼─────────────┐
                │   编排者 Qwen3-8B（一次加载） │ ← 规则路由(agent.py) 或
                │   决策：调哪个能力 Agent      │   模型自主路由(function-calling)
                └─────────────┬─────────────┘
           ┌─────────┬─────────┼─────────┬─────────┬─────────┐
           ▼         ▼         ▼         ▼         ▼         ▼
       文本检索    数字查表    影像解读   变化检测    知识融合    报告生成
      BM25+向量   metric_    Qwen3-VL  VLM 双时相  RAG证据+   变化图标红
      +rerank    lookup(零LLM)          T1/T2     影像→VLM   +指标→五段报告
           └─────────┴─────────┴─────────┴─────────┴─────────┘
                              │
                    FastMCP 对外暴露（stdio，mcp_server.py）
```

## 核心数字（全部为服务器对账后口径）

| 维度 | 结果 |
|------|------|
| 语料 | 120 篇 arXiv 论文（~68 train / ~52 heldout，SHA256 确定性切分，**留出集召回 0% 实证物理隔离**） |
| 检索 | BM25 + bge-m3 双路各召回 32 → 并集 ≤64 → CrossEncoder 精排 top-4；索引四代演进定版 180 字 / 40 重叠 |
| 数字忠实率 | 纯向量基线 17.4% → 确定性查表通道 **89.7%** → 基座升级 Qwen3-8B **97.2%**（450 题集） |
| 引用率 | 基线 13.6% → SFT v2 **56.6%** → Qwen3-8B **98.2%** |
| 对齐训练 | 走完 SFT→merge→DPO→GRPO 全链路；DPO 负收益、GRPO 收益未显现，**均经对照实验定位根因并如实记录** |
| 最大对照发现 | 同一套 RAG 仅换基座（Qwen3-8B）即**全面超过整条微调线**——对齐技巧替代不了基座能力 |
| 误差分账 | Oracle 对照 + 三指标分离：名称题瓶颈在生成端、数字题在检索端，后续投入按分账定向分配 |
| 多 Agent 并行 | 图型调度（依赖任务板 + 邮箱直连 + 线程池）自测 2×0.5s 任务 **0.51s** 完成 |
| AI 产出审计 | 双独立审阅者核对 **199 条声明** → 修复 13 处偏差 + 抓出 1 个真实代码 bug |

## 四大模块

### 1️⃣ RAG：混合检索 + 数字确定性通道

- **双路互补召回**：BM25 稀疏检索（数字/专有名词强）+ bge-m3 稠密检索（语义改写强），并集后 CrossEncoder 精排；生成端三约束 + 贪心解码保证可复现。
- **数字不过模型**：针对数值幻觉设计确定性查表通道——离线抽取「指标-数字」对（`build_metric_lookup.py`，8657 条四元组指标库）并回溯定位原文句子，在线命中即查表作答，LLM 全程不回忆数字；未命中三级降级回检索，永不空手。
- **防泄漏**：评测题与检索索引按论文 SHA256 哈希物理隔离，留出集实测召回 0%。
- 后期演进：Docling 表格抽取、与 PaperQA2 框架同题对照（109 篇全文 + 50 概念题）。

### 2️⃣ RL：对齐全链路 + 诚实负结果

- 程序化生成 SFT 数据与 DPO 偏好对（`train_data_gen_v2.py`），训练全程 13 道工程防线（输入哈希、确定性切分、命令留档可重放、产物断言）。
- SFT 将引用率 13.6% → 56.6%（`run_rscd_sft*.sh`）；DPO 负收益如实归因（oracle 引用 67.3% vs 真实检索崩至 0.5% 的撕裂 = 学偏铁证）；GRPO 用规则 verifier 与评测同口径（`grpo_training_rscd.py`），小步数下收益未显现。
- **基座对照是全项目最值钱的一行数据**：仅换 Qwen3-8B 即全面超过微调线（引用 98.2%）。微调线的价值在受控实验与误差归因的方法论。

### 3️⃣ 多 Agent 协同：星型主干 + 图型升级

- 编排者–工作者架构：Qwen3-8B 单次加载兼任编排与生成，调度 6 类能力 Agent（`agent_toolcall.py` / `multi.py`）。
- 本地推理无原生工具调用通道，**function-calling 协议自实现**（schema 注入、双正则解析、role:"tool" 回传、失败四重兜底），经 **FastMCP** 对外标准化暴露（`mcp_server.py`）。
- 图型升级：依赖任务板状态机 + Agent 间邮箱直连 + 线程池并行；能力插件化（`plugin_*.py` / `catalog.py`，新增能力仅需注册一条，调度器零改动）。

### 4️⃣ 评测与审阅：误差分账 + 独立审计

- 自建三代评测集（铁律「检索只见题目、gold 答案只用于打分」）；**Oracle 对照 + DocRec/EvRec/AnsAcc 三指标分离**做误差分账，同一套指标分出两种病、开两种药。
- RAGAS + RAGChecker 双框架交叉验证（`eval_run*.py` / `run_judge_*.sh`）；自建 LLM-as-judge 三维判分（grounded/relevance/honesty）七轮迭代，bootstrap 95% 置信区间管理评测偶然性。
- **双独立审阅者工作流**审计 AI 产出：写审分离、清单前置、三值判定 + 行号证据——199 条声明核对、13 处修复、1 个真实代码 bug。

## 目录导览

| 文件 | 说明 |
|------|------|
| `rsagent_web.py` | Flask 单文件 Web 聊天界面（端口 7861，主入口） |
| `agent_toolcall.py` / `agent.py` | 6 工具 function-calling 完整版 Agent / 规则路由版 |
| `mcp_server.py` | MCP server（stdio），对外暴露 4 个标准工具 |
| `rag_pipeline.py` / `embedding_rerank.py` | RAG 检索链路 / 嵌入与精排 |
| `build_*.py` | 语料、索引、指标库、题库构建脚本 |
| `eval_*.py` / `run_judge_*.sh` | 三代评测集构建与判分 |
| `grpo_training_rscd.py` / `run_rscd_*.sh` | SFT/DPO/GRPO 训练 |
| `plugin_*.py` / `catalog.py` | 能力插件化框架 |
| `persona_*.py` / `mentor*.py` | mentor 教学蒸馏探索（小规模，如实标注） |
| `qwen3vl_*.py` | 影像解读 / 双时相变化检测（Qwen3-VL） |
| `*.jsonl` / `*.json` | 评测集、答案、评测报告与指标库数据 |

## 快速开始

```bash
# 1. 环境：Python 3.10+，主要依赖
pip install flask llama-index faiss-cpu rank-bm25 jieba pymupdf \
            sentence-transformers vllm peft trl fastmcp openai

# 2. 模型：Qwen3-8B / Qwen3-VL-8B / bge-m3 / bge-reranker-v2-m3（见 download_models.py）

# 3. 构建索引与指标库（论文语料自备，见下方说明）
python build_llama_index.py && python build_metric_lookup.py

# 4. 启动 Web 界面（vLLM 服务就绪后）
python rsagent_web.py   # → http://127.0.0.1:7861

# 或作为 MCP 工具供任意 MCP 客户端调用
python mcp_server.py
```

> 环境变量：第三方 API 走 `OPENAI_API_KEY` / `OPENAI_API_BASE`（代码中为占位符 `<YOUR_API_KEY>`，请自行填入）。

## 诚实边界（主动交代）

- 图型调度三模块为**自测通过**（模拟任务），尚未接真机能力；
- DPO / GRPO 为**负结果**，已如实记录并归因——SFT 的引用率提升是真实正收益；
- 引用率指标只测 `[n]` 出现，不验证引用内容是否真正支撑答案；
- 论文全文语料（120 篇 arXiv PDF/txt）因版权与体积原因**不入库**，`papers_manifest.json` 提供清单，可按 manifest 自行构建。

## License

个人学习研究项目，代码供学习交流。
