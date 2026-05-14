# Industrial PDF RAG Demo

## Quick Start

最上面的演示入口现在是 `main.py`，它会直接复用当前已经打通的在线检索和回答链。
默认是终端流式输出最终答案，并在“引用来源”后追加对应的本地页图路径和证据文本，方便演示时直接定位手册页。

当前版本只是一个构架思路的验证，模型的兜底和全链路的追踪均未实现。
链条中的部分节点采用手工搓的代码实现，只是用于验证构架思路，请勿用于生产环境。
生产环境中需要重新评估链接、模型、数据集、工具等，并重新实现。
如：DashScope OCR fix 节点 需要引用PaddleOCR结构 + Qwen2-VL内容混合模式，Knowledge Graph 节点 需要引入Neo4j 
    


### 1. 安装 UV

如果本机还没有 `uv`，先执行：

```powershell
powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

安装完成后重新打开一个 PowerShell 窗口。

### 2. 安装依赖

在项目根目录执行：

```powershell
uv sync
```

### 3. 检查 `.env`

当前 DEMO 依赖以下配置：

```text
OPENAI_API_KEY=DS的KEY
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_MODEL=deepseek-chat

# 阿里
DASHSCOPE_API_KEY=阿里百炼的KEY
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_MODEL=text-embedding-v2
```

### 4. 运行聊天 DEMO

默认会加载样例文档 `300220_FANUC0iMF` 的现成检索块、KB 和 KG：

```powershell
uv run python .\main.py
```

如果 Windows 终端里中文显示乱码，先执行：

```powershell
chcp 65001
```

如果要显式指定输入文件：

```powershell
uv run python .\main.py `
  --final-blocks .\data\blocks\300220_FANUC0iMF\final_blocks_ocr_fixed.jsonl `
  --kb-blocks .\data\kb\300220_FANUC0iMF\kb_blocks.jsonl `
  --kb-graph .\data\kb\300220_FANUC0iMF\kb_graph.json `
  --top-k 5
```

### 5. 演示用法

先直接提问：

```text
SP9137 主轴通信异常怎么办？
```

如果回答里的第 7 项要求补充现场信息，就继续输入：

```text
补充信息：主轴放大器 LED 红灯，上电即报警，同时没有其他 SP/SV 报警
```

脚本会基于“上一轮问题 + 补充信息”重新走一遍：

```text
Query
-> Parse
-> Retrieval
-> Merge
-> Rerank
-> TopK Evidence Package
-> Rule-based Reasoning Planner
-> Evidence-grounded Answer Prompt
-> LLM Answer
-> Validate
-> Final Answer
```

支持的命令：

```text
/reset  清空上一轮上下文，并重新开始新提问
/exit   退出
```

当前版本目标是把工业手册从 `PDF -> Retrieval -> Reasoning -> Answer -> Validate` 这条 DEMO 链路跑通。

当前主流程：

```text
PDF
-> Docling layout / page images / layout.json
-> draft_blocks / final_blocks
-> garbled page detect
-> DashScope OCR fix
-> final_blocks_ocr_fixed
-> Knowledge Blocks
-> Knowledge Graph
-> Query -> Parse -> Retrieval -> Merge -> Rerank -> TopK Evidence Package
-> Rule-based Reasoning Planner
-> Evidence-grounded Answer Prompt
-> LLM Answer
-> Validate
-> Final Answer
```

## 目录

```text
data/
  raw/       # 原始 PDF
  parsed/    # Docling Markdown / layout.json
  images/    # 每页 PNG
  blocks/    # blocks.jsonl
  index/     # 本地 embedding 缓存
  eval/      # 检索评测用例和报告
scripts/
  common.py
  01_parse_pdf_light.py
  01_parse_pdf.py
  02_build_blocks.py       # writes draft_blocks.jsonl and final_blocks.jsonl
  03_search_demo.py
  04_eval_retrieval.py
  05_detect_missing_coverage.py
  06_build_refine_queue.py
  07_detect_refine_pages.py
  08_parse_refine_pages.py
  09_merge_blocks.py
  10_search_llamaindex.py
  11_validate_rag_pipeline.py
  llamaindex_hybrid_pipeline.py
  search_loader.py
  refine_pages_loader.py
```

## 安装

```powershell
uv sync
```

## 运行

当前 PDF 已放在：

```text
data/raw/300220、FANUC0iMF维修说明书.pdf
```

解析 PDF：

```powershell
uv run python scripts/01_parse_pdf_light.py "data/raw/300220、FANUC0iMF维修说明书.pdf"
```

命令会打印规范化后的 `doc_id`。当前文件默认会生成：

```text
300220_FANUC0iMF
```

构建 blocks：

```powershell
uv run python scripts/02_build_blocks.py 300220_FANUC0iMF
```

单次检索：

```powershell
uv run python scripts/03_search_demo.py 300220_FANUC0iMF "SP9137 DEVICE COMMUNICATION ERROR" --retriever bm25
uv run python scripts/03_search_demo.py 300220_FANUC0iMF "SP9137 DEVICE COMMUNICATION ERROR" --retriever hybrid
```

批量评测：

```powershell
uv run python scripts/04_eval_retrieval.py 300220_FANUC0iMF data/eval/300220_FANUC0iMF/eval_queries.jsonl --retriever bm25
uv run python scripts/04_eval_retrieval.py 300220_FANUC0iMF data/eval/300220_FANUC0iMF/eval_queries.jsonl --retriever hybrid
```

评测会输出 `recall@1`、`recall@5`、`recall@10`，并写入：

```text
data/eval/{doc_id}/retrieval_report.json
data/eval/{doc_id}/failure_report.json
data/eval/{doc_id}/retrieval_report_bm25.json
data/eval/{doc_id}/failure_report_bm25.json
data/eval/{doc_id}/retrieval_report_hybrid.json
data/eval/{doc_id}/failure_report_hybrid.json
```

## 第二遍精解析

第一遍是全书覆盖，第二遍只处理重点页。先检测候选页：

```powershell
uv run python scripts/07_detect_refine_pages.py 300220_FANUC0iMF
```

如果你知道某页是表格/报警密集页，可以手动指定，比如第 622 页：

```powershell
uv run python scripts/07_detect_refine_pages.py 300220_FANUC0iMF --pages 622
```

对候选页跑精解析：

```powershell
uv run python scripts/08_parse_refine_pages.py 300220_FANUC0iMF
```

也可以只跑指定页验证：

```powershell
uv run python scripts/08_parse_refine_pages.py 300220_FANUC0iMF --pages 622
```

合并成最终检索 blocks：

```powershell
uv run python scripts/09_merge_blocks.py 300220_FANUC0iMF
```

合并规则：

```text
有 refined_blocks 的页：
  用 refined alarm/table blocks 替换该页 draft alarm_candidate/table_candidate
  保留普通 text/procedure/warning/image blocks 作为补充

没有 refined_blocks 的页：
  继续使用 draft blocks
```

检索始终默认读取：

```text
data/blocks/{doc_id}/final_blocks.jsonl
```

代码检查：

```powershell
uv run python -m compileall scripts
```

## LlamaIndex + Chroma RAG

`final_blocks.jsonl` 会补齐这些字段，供检索和展示直接使用：

```text
created_at, updated_at
context_prev, context_next
page_context_id
```

运行向量检索和重排前需要先设置 DashScope Key：

```powershell
$env:DASHSCOPE_API_KEY = "your_key"
```

可选模型环境变量：

```powershell
$env:DASHSCOPE_EMBEDDING_MODEL = "text-embedding-v4"
$env:DASHSCOPE_RERANK_MODEL = "qwen3-rerank"
```


单条查询：

```powershell
uv run python scripts/10_search_llamaindex.py data/blocks/300220_FANUC0iMF/final_blocks.jsonl "SP9137 DEVICE COMMUNICATION ERROR" --retriever hybrid_rerank
```

批量验证：

```powershell
uv run python scripts/11_validate_rag_pipeline.py data/blocks/300220_FANUC0iMF/final_blocks.jsonl data/eval/300220_FANUC0iMF/eval_queries.jsonl
```

## Knowledge Blocks

```powershell
uv run python scripts/12_build_kb_blocks.py data/blocks/300220_FANUC0iMF/final_blocks.jsonl
```

```text
output: data/kb/{doc_id}/kb_blocks.jsonl
split: alarm -> per code, procedure -> per step, table -> per row, text/warning -> per paragraph
```

## Knowledge Graph

```powershell
uv run python scripts/13_build_kb_graph.py data/kb/300220_FANUC0iMF/kb_blocks.jsonl
```

```text
output: data/kb/{doc_id}/kb_graph.json
nodes: kb + entity
edges: mentions / triggers / depends_on / adjacent / same_page / related_context
```

## Demo Pipeline

当前完整 DEMO 链路：

```text
PDF / existing refined blocks
-> final_blocks
-> garbled page detect
-> DashScope OCR
-> final_blocks_ocr_fixed
-> Knowledge Blocks
-> Knowledge Graph
-> BM25 / Hybrid Rerank / Graph Expanded Rerank validation
```

脚本顺序和对应关系：

```text
01_parse_pdf.py
  PDF -> page images + layout.json + pages_text.jsonl

02_build_blocks.py
  layout.json -> draft_blocks.jsonl / final_blocks.jsonl

09_merge_blocks.py
  draft_blocks + refined_blocks -> final_blocks.jsonl
  如果 data/refine/{doc_id}/refined_blocks.jsonl 已存在，优先合并历史精解析结果

14_detect_garbled_pages.py
  final_blocks.jsonl -> data/ocr/{doc_id}/garbled_pages.json

15_ocr_pages_dashscope.py
  garbled page images -> data/ocr/{doc_id}/ocr_pages.jsonl

16_fix_blocks_with_ocr.py
  final_blocks.jsonl + ocr_pages.jsonl -> final_blocks_ocr_fixed.jsonl

12_build_kb_blocks.py
  final_blocks_ocr_fixed.jsonl -> data/kb/{doc_id}/kb_blocks.jsonl

13_build_kb_graph.py
  kb_blocks.jsonl -> data/kb/{doc_id}/kb_graph.json

11_validate_rag_pipeline.py
  final_blocks_ocr_fixed.jsonl (+ kb_blocks.jsonl + kb_graph.json)
  -> rag_pipeline_validation.json
```

推荐执行顺序：

```powershell
uv run python scripts/14_detect_garbled_pages.py data/blocks/300220_FANUC0iMF/final_blocks.jsonl
uv run python scripts/15_ocr_pages_dashscope.py 300220_FANUC0iMF
uv run python scripts/16_fix_blocks_with_ocr.py data/blocks/300220_FANUC0iMF/final_blocks.jsonl data/ocr/300220_FANUC0iMF/ocr_pages.jsonl
uv run python scripts/12_build_kb_blocks.py data/blocks/300220_FANUC0iMF/final_blocks_ocr_fixed.jsonl
uv run python scripts/13_build_kb_graph.py data/kb/300220_FANUC0iMF/kb_blocks.jsonl
uv run python scripts/11_validate_rag_pipeline.py data/blocks/300220_FANUC0iMF/final_blocks_ocr_fixed.jsonl data/eval/300220_FANUC0iMF/eval_queries.jsonl --retrievers bm25 hybrid_rerank graph_expanded_rerank --kb-blocks-jsonl data/kb/300220_FANUC0iMF/kb_blocks.jsonl --kb-graph-json data/kb/300220_FANUC0iMF/kb_graph.json
```

一键跑完整 DEMO：

```powershell
uv run python scripts/17_run_demo_pipeline.py "data/raw/300220、FANUC0iMF维修说明书.pdf" --skip-parse --doc-id 300220_FANUC0iMF --validate --retrievers bm25 hybrid_rerank graph_expanded_rerank
```

说明：

```text
17_run_demo_pipeline.py 默认会尽量复用已有的 final_blocks / final_blocks_ocr_fixed / kb_blocks / kb_graph
只有输入文件更新或显式传 --force 时才重建对应阶段
适合后续换同类型 PDF 做快速 DEMO 验证
```

## Block Schema

每个 block 包含：

```text
block_id, doc_id, block_type, block_level, source_priority, page_no
title, raw_text, canonical_text, searchable_text, alarm_codes
section_path, hierarchy, prev_block_id, next_block_id, parent_block_id
layout_refs, bbox, image_paths, has_image, ocr_confidence, source
needs_refine, refine_reason, parse_level
```

第一版保留整页图片，不做局部裁剪；`bbox`、`ocr_confidence` 从 Docling JSON 中 best-effort 提取。

`02_build_blocks.py` 会先生成 light parse 的 `draft_blocks.jsonl`，并在没有 refined blocks 的第一版中同步生成 `final_blocks.jsonl`。后续第二遍精解析只需要写入 `refined_blocks.jsonl`，再按页合并覆盖 candidate blocks 即可。

## Online Evidence Retrieval

在线查询链路现在固定为：

```text
Query
-> Rule-based Query Parse
-> Entity Match + BM25 + Vector + Graph Expansion
-> Merge / Dedup / Score Fusion
-> Qwen3-Reranker
-> TopK Evidence Package
```

单条查询命令：

```powershell
uv run python scripts/18_search_evidence_package.py data/blocks/300220_FANUC0iMF/final_blocks_ocr_fixed.jsonl data/kb/300220_FANUC0iMF/kb_blocks.jsonl data/kb/300220_FANUC0iMF/kb_graph.json "SP9137 主轴通信异常怎么办？" --top-k 5 --out data/eval/300220_FANUC0iMF/evidence_package_sp9137.json
```

输出：

```text
data/eval/{doc_id}/evidence_package_*.json
```

说明：

```text
18_search_evidence_package.py 会输出 TopK Evidence Package，不负责最终答案生成
Block Vector Retrieval 走 Chroma，本地命中已建索引时不会重新为整本 final_blocks 生成 embedding
只有 final_blocks 文件变更、embedding 模型变更，或显式删除 data/index/{doc_id}/llamaindex_chroma 后，才需要重建 block 向量索引
当前 KB / Graph 层仍是本地内存结构，不是独立的 Chroma collection
对 SP9137 / 参数号 / PLC 地址 这类强结构化 query，会自动跳过最慢的向量召回，优先走 entity + bm25 + graph + rerank 快路径
```

建议：

```text
首次处理新文档：先跑 17_run_demo_pipeline.py，把 final_blocks_ocr_fixed / kb_blocks / kb_graph / block index 都准备好
后续在线查同一文档：直接跑 18_search_evidence_package.py，耗时应主要是 query embedding + rerank，而不是整库重建
```

在线链路阶段职责：

```text
Query: 用户原始问题
Rule-based Query Parse: 提取 alarm_codes / entities / intent / query_type / expanded_queries
Retrieval: Entity Match + BM25 + Vector + Graph Expansion
Merge: 合并去重并做初排分数融合
Rerank: Qwen3-Reranker + 轻量业务偏置
TopK Evidence Package: 输出给后续 Reasoning / Answer 使用的证据包
```

20 条验收命令：

```powershell
uv run python scripts/19_validate_online_evidence.py data/blocks/300220_FANUC0iMF/final_blocks_ocr_fixed.jsonl data/kb/300220_FANUC0iMF/kb_blocks.jsonl data/kb/300220_FANUC0iMF/kb_graph.json data/eval/300220_FANUC0iMF/online_evidence_eval_queries.jsonl --out data/eval/300220_FANUC0iMF/online_evidence_validation.json
```

验收样本：

```text
data/eval/{doc_id}/online_evidence_eval_queries.jsonl
共 20 条，覆盖 alarm / semantic / procedure / safety 四类 query
```

验收输出：

```text
data/eval/{doc_id}/online_evidence_validation.json
```

当前样本文档结果：

```text
alarm query: Top1 命中正确报警 KB，8/8
无报警码语义 query: Top5 命中相关 KB，5/5
操作步骤 query: Top3 包含 procedure KB，3/3
安全类 query: Top3 包含 warning KB，4/4
Graph 扩展: 需要 graph 的 case 均能带出 graph 来源候选
Rerank: 当前样本上对 merge 初排不降级，procedure 类有 1 条从 merge 非 Top1 被 rerank 提升到 Top1
```

## Reasoning And Answer

当前回答链路：

```text
TopK Evidence Package
-> Rule-based Reasoning Planner
-> Evidence-grounded Answer Prompt
-> LLM Answer
-> Answer Validate
-> Final Answer
```

职责：

```text
Reasoning: 判断证据是否足够、给证据分组、选主证据/辅助证据/安全证据、生成 answer_plan
Answer: 基于 reasoning + evidence 生成最终 JSON 回答
Validate: 校验 section、页码、KB 引用、安全提醒、证据不足说明、未见报警码越界
```

脚本顺序：

```text
18_search_evidence_package.py
  Query -> Parse -> Retrieval -> Merge -> Rerank -> TopK Evidence Package

20_build_reasoning.py
  Evidence Package -> Rule-based Reasoning Planner

21_generate_answer.py
  Evidence Package + Reasoning -> LLM Answer -> Validate -> Final Answer
```

命令示例：

```powershell
uv run python scripts/18_search_evidence_package.py data/blocks/300220_FANUC0iMF/final_blocks_ocr_fixed.jsonl data/kb/300220_FANUC0iMF/kb_blocks.jsonl data/kb/300220_FANUC0iMF/kb_graph.json "SP9137" --top-k 5 --out data/eval/300220_FANUC0iMF/evidence_package_sp9137_ascii.json

uv run python scripts/20_build_reasoning.py data/eval/300220_FANUC0iMF/evidence_package_sp9137_ascii.json --out data/eval/300220_FANUC0iMF/reasoning_sp9137_ascii.json

uv run python scripts/21_generate_answer.py data/eval/300220_FANUC0iMF/evidence_package_sp9137_ascii.json --reasoning-json data/eval/300220_FANUC0iMF/reasoning_sp9137_ascii.json --out data/eval/300220_FANUC0iMF/answer_sp9137_ascii.json
```

说明：

```text
Reasoning 当前是规则化实现，不依赖 LLM
Answer 当前使用 .env 中配置的 OPENAI_BASE_URL / OPENAI_MODEL / OPENAI_API_KEY，通过 openai SDK 调用 OpenAI 兼容接口
如果 LLM 输出未通过 Validate，会自动降级到模板化回答
```
