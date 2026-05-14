from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from llamaindex_hybrid_pipeline import DEFAULT_EMBEDDING_MODEL, DEFAULT_RERANK_MODEL
from online_evidence_retriever import load_online_evidence_retriever
from reasoning_answer_pipeline import build_final_answer, build_reasoning


DEFAULT_DOC_ID = "300220_FANUC0iMF"
DEFAULT_FINAL_BLOCKS = PROJECT_ROOT / "data" / "blocks" / DEFAULT_DOC_ID / "final_blocks_ocr_fixed.jsonl"
DEFAULT_KB_BLOCKS = PROJECT_ROOT / "data" / "kb" / DEFAULT_DOC_ID / "kb_blocks.jsonl"
DEFAULT_KB_GRAPH = PROJECT_ROOT / "data" / "kb" / DEFAULT_DOC_ID / "kb_graph.json"
SUPPLEMENT_PREFIXES = ("补充信息", "补充：", "补充:", "用户：", "用户:", "现场补充", "补充说明")
STREAM_CHUNK_CHARS = 96


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive CNC manual QA demo.")
    parser.add_argument("--final-blocks", default=str(DEFAULT_FINAL_BLOCKS))
    parser.add_argument("--kb-blocks", default=str(DEFAULT_KB_BLOCKS))
    parser.add_argument("--kb-graph", default=str(DEFAULT_KB_GRAPH))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--rerank-model", default=DEFAULT_RERANK_MODEL)
    parser.add_argument("--show-debug", action="store_true")
    return parser.parse_args()


def configure_stdio() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def ensure_inputs_exist(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required input files:\n" + "\n".join(missing))


def is_supplement_input(text: str) -> bool:
    normalized = text.strip()
    return any(normalized.startswith(prefix) for prefix in SUPPLEMENT_PREFIXES)


def strip_supplement_prefix(text: str) -> str:
    normalized = text.strip()
    for prefix in SUPPLEMENT_PREFIXES:
        if normalized.startswith(prefix):
            remainder = normalized[len(prefix) :].lstrip("：: ").strip()
            return remainder or normalized
    return normalized


def build_query(base_query: str, supplements: list[str]) -> str:
    if not supplements:
        return base_query.strip()
    lines = [base_query.strip(), "", "补充信息："]
    lines.extend(f"- {item}" for item in supplements if item.strip())
    return "\n".join(lines).strip()


def resolve_local_path(path_str: str) -> str:
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    return str((PROJECT_ROOT / path).resolve())


def format_value(value: Any, indent: str = "") -> list[str]:
    if isinstance(value, str):
        text = value.strip()
        return [f"{indent}{text}"] if text else [f"{indent}(空)"]
    if isinstance(value, (int, float)):
        return [f"{indent}{value}"]
    if isinstance(value, dict):
        lines: list[str] = []
        for key, item in value.items():
            if isinstance(item, (list, dict)):
                lines.append(f"{indent}- {key}:")
                lines.extend(format_value(item, indent + "  "))
            else:
                lines.append(f"{indent}- {key}: {item}")
        return lines or [f"{indent}(空对象)"]
    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, dict):
                summary_parts = []
                if item.get("alarm_code"):
                    summary_parts.append(f"alarm_code={item['alarm_code']}")
                if item.get("page_no") is not None:
                    summary_parts.append(f"page_no={item['page_no']}")
                if item.get("section_path"):
                    summary_parts.append(f"section_path={item['section_path']}")
                if item.get("kb_id"):
                    summary_parts.append(f"kb_id={item['kb_id']}")
                if summary_parts:
                    lines.append(f"{indent}- " + ", ".join(summary_parts))
                else:
                    lines.append(f"{indent}- {json.dumps(item, ensure_ascii=False)}")
            else:
                lines.append(f"{indent}- {item}")
        return lines or [f"{indent}(空列表)"]
    return [f"{indent}{value}"]


def collect_reasoning_evidence(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    evidence_by_kb: dict[str, dict[str, Any]] = {}
    for bucket_name in ("main_evidence", "supporting_evidence", "safety_evidence"):
        for item in result["reasoning"].get(bucket_name) or []:
            kb_id = str(item.get("kb_id") or "").strip()
            if kb_id and kb_id not in evidence_by_kb:
                evidence_by_kb[kb_id] = item
    return evidence_by_kb


def render_local_evidence_files(result: dict[str, Any]) -> list[str]:
    citations = result["answer"].get("citations") or []
    evidence_by_kb = collect_reasoning_evidence(result)
    lines: list[str] = []
    seen: set[tuple[int | None, str]] = set()
    for citation in citations:
        kb_id = str(citation.get("kb_id") or "").strip()
        if not kb_id:
            continue
        item = evidence_by_kb.get(kb_id)
        if not item:
            continue
        key = (item.get("page_no"), kb_id)
        if key in seen:
            continue
        seen.add(key)
        lines.append(
            f"- page_no={item.get('page_no')}, kb_id={kb_id}, block_id={item.get('block_id')}"
        )
        if item.get("section_path"):
            lines.append(f"  section_path={item['section_path']}")
        for image_path in item.get("image_paths") or []:
            lines.append(f"  image={resolve_local_path(str(image_path))}")
        text = str(item.get("text") or "").strip()
        if text:
            lines.append(f"  text={text[:200]}")
    return lines


def render_final_answer(result: dict[str, Any]) -> str:
    reasoning = result["reasoning"]
    answer_obj = result["answer"]["answer"]
    sections = reasoning["answer_plan"]["sections"]
    lines = [
        "",
        "=" * 88,
        f"问题: {result['query']}",
        f"回答类型: {reasoning['answer_type']}",
        f"能否直接回答: {'是' if reasoning['can_answer'] else '否'}",
        f"置信度: {reasoning['confidence']}",
        f"生成模式: {result['generation_mode']}",
        "=" * 88,
    ]
    for index, section in enumerate(sections, start=1):
        lines.append(f"{index}. {section}")
        lines.extend(format_value(answer_obj.get(section, "")))
        lines.append("")
    citations = result["answer"].get("citations") or []
    if citations:
        lines.append("引用来源")
        lines.extend(format_value(citations))
        lines.append("")
    local_evidence_lines = render_local_evidence_files(result)
    if local_evidence_lines:
        lines.append("本地证据文件")
        lines.extend(local_evidence_lines)
        lines.append("")
    if result.get("validation"):
        validation = result["validation"]
        lines.append(
            "校验结果: "
            f"{'通过' if validation.get('ok') else '未通过'}"
        )
    return "\n".join(lines).strip()


def print_streamed_text(text: str) -> None:
    for start in range(0, len(text), STREAM_CHUNK_CHARS):
        chunk = text[start : start + STREAM_CHUNK_CHARS]
        sys.stdout.write(chunk)
        sys.stdout.flush()
        time.sleep(0.01)
    if not text.endswith("\n"):
        sys.stdout.write("\n")
        sys.stdout.flush()


def print_intro() -> None:
    print("输入问题后直接回车即可，例如：SP9137 主轴通信异常怎么办？")
    print("如果上一轮答案要求补充现场信息，可以继续输入：补充信息：主轴放大器 LED 红灯")
    print("命令：/reset 重新开始新提问，/exit 退出")


def main() -> None:
    configure_stdio()
    args = parse_args()
    final_blocks = Path(args.final_blocks)
    kb_blocks = Path(args.kb_blocks)
    kb_graph = Path(args.kb_graph)
    ensure_inputs_exist([final_blocks, kb_blocks, kb_graph])

    retriever = load_online_evidence_retriever(
        final_blocks,
        kb_blocks,
        kb_graph,
        embedding_model=args.embedding_model,
        rerank_model=args.rerank_model,
    )

    last_question = ""
    supplements: list[str] = []

    print("Industrial RAG Chat Demo")
    print(f"final_blocks: {final_blocks}")
    print(f"kb_blocks:    {kb_blocks}")
    print(f"kb_graph:     {kb_graph}")
    print_intro()

    while True:
        try:
            user_input = input("\n用户> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            break

        if not user_input:
            continue
        if user_input.lower() in {"/exit", "exit", "quit"}:
            print("已退出。")
            break
        if user_input.lower() == "/reset":
            last_question = ""
            supplements = []
            print("已清空上一轮上下文，请直接输入新问题。")
            continue

        if is_supplement_input(user_input):
            if not last_question:
                print("还没有上一轮问题，先输入一个主问题。")
                continue
            supplement = strip_supplement_prefix(user_input)
            if supplement:
                supplements.append(supplement)
            query = build_query(last_question, supplements)
        else:
            last_question = user_input
            supplements = []
            query = last_question

        print("\n助手> 正在检索证据并生成答案，请稍候...\n")
        evidence_package = retriever.retrieve(query, top_k=args.top_k)
        reasoning = build_reasoning(evidence_package)
        final_answer = build_final_answer(evidence_package, reasoning)

        print_streamed_text(render_final_answer(final_answer))

        if args.show_debug:
            print("\n调试信息")
            print(json.dumps(evidence_package["parsed_query"], ensure_ascii=False, indent=2))
            print(json.dumps(evidence_package["debug"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
