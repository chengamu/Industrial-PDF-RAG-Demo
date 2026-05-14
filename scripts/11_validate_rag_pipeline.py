import argparse
from pathlib import Path
from typing import Any, Dict, List

from common import DATA_DIR, canonicalize_text, ensure_dir, normalize_alarm_code, read_jsonl, write_json
from llamaindex_hybrid_pipeline import DEFAULT_EMBEDDING_MODEL, DEFAULT_RERANK_MODEL, load_pipeline
from kb_graph_retriever import load_kb_graph_retriever


def normalize_codes(codes: List[str]) -> List[str]:
    return sorted(set(normalize_alarm_code(code) for code in codes if code))


def block_text(block: Dict[str, Any]) -> str:
    return canonicalize_text(
        "\n".join(
            [
                str(block.get("title") or ""),
                " ".join(block.get("alarm_codes") or []),
                str(block.get("section_path") or ""),
                str(block.get("searchable_text") or ""),
                str(block.get("raw_text") or ""),
            ]
        )
    )


def expected_codes_hit(blocks: List[Dict[str, Any]], expected_codes: List[str]) -> bool:
    if not expected_codes:
        return False
    expected_set = set(expected_codes)
    for block in blocks:
        block_codes = set(normalize_codes(block.get("alarm_codes") or []))
        if block_codes.intersection(expected_set):
            return True
        text = block_text(block)
        if any(code in text for code in expected_codes):
            return True
    return False


def expected_pages_hit(blocks: List[Dict[str, Any]], expected_pages: List[int]) -> bool:
    return bool(expected_pages and set(block.get("page_no") for block in blocks).intersection(set(expected_pages)))


def expected_keywords_hit(blocks: List[Dict[str, Any]], expected_keywords: List[str]) -> bool:
    if not expected_keywords:
        return False
    joined = "\n".join(block_text(block) for block in blocks)
    return all(canonicalize_text(keyword) in joined for keyword in expected_keywords)


def summarize_result(item: Dict[str, Any]) -> Dict[str, Any]:
    hit = item["hit_block"]
    return {
        "score": item.get("score"),
        "score_detail": item.get("score_detail"),
        "block_id": hit.get("block_id") or hit.get("kb_id"),
        "block_type": hit.get("block_type") or hit.get("kb_type"),
        "page_no": hit.get("page_no"),
        "alarm_codes": hit.get("alarm_codes") or hit.get("entities") or [],
        "section_path": hit.get("section_path"),
        "page_image": item.get("page_image"),
        "text_preview": str(hit.get("raw_text") or hit.get("text") or "")[:240],
        "context_block_ids": [block.get("block_id") or block.get("kb_id") for block in (item.get("context_blocks") or [])],
        "page_context_block_id": (item.get("page_context") or {}).get("block_id") or (item.get("page_context") or {}).get("page_context_id"),
        "graph_neighbors": item.get("graph_neighbors") or [],
    }


def evaluate_case(pipeline, case: Dict[str, Any], retriever: str) -> Dict[str, Any]:
    query = str(case.get("query") or "")
    top_k = int(case.get("top_k") or 10)
    results = pipeline.search(query, retriever=retriever, top_k=top_k)
    blocks = [item["hit_block"] for item in results]

    expected_alarm_codes = normalize_codes(case.get("expected_alarm_codes") or [])
    expected_pages = [int(page) for page in (case.get("expected_pages") or [])]
    expected_keywords = [str(keyword) for keyword in (case.get("expected_keywords") or []) if str(keyword)]

    alarm_hit = expected_codes_hit(blocks, expected_alarm_codes) if expected_alarm_codes else False
    page_hit = expected_pages_hit(blocks, expected_pages) if expected_pages else False
    keyword_hit = expected_keywords_hit(blocks, expected_keywords) if expected_keywords else False
    overall_hit = any([alarm_hit, page_hit, keyword_hit]) if any([expected_alarm_codes, expected_pages, expected_keywords]) else False

    score_values = [float(item.get("score") or 0.0) for item in results]
    rerank_scores = [
        float((item.get("score_detail") or {}).get("rerank_score") or 0.0)
        for item in results
        if "rerank_score" in (item.get("score_detail") or {})
    ]
    schema_ok = all(
        {"score", "hit_block", "context_blocks", "page_context", "page_image"}.issubset(set(item.keys()))
        for item in results
    )

    return {
        "query": query,
        "retriever": retriever,
        "top_k": top_k,
        "expected_alarm_codes": expected_alarm_codes,
        "expected_pages": expected_pages,
        "expected_keywords": expected_keywords,
        "alarm_hit": alarm_hit,
        "page_hit": page_hit,
        "keyword_hit": keyword_hit,
        "overall_hit": overall_hit,
        "result_count": len(results),
        "schema_ok": schema_ok,
        "score_stats": {
            "max_score": max(score_values) if score_values else 0.0,
            "min_score": min(score_values) if score_values else 0.0,
            "avg_score": (sum(score_values) / len(score_values)) if score_values else 0.0,
            "avg_rerank_score": (sum(rerank_scores) / len(rerank_scores)) if rerank_scores else None,
        },
        "topk": [summarize_result(item) for item in results],
    }


def summarize_cases(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(cases)
    if total == 0:
        return {
            "query_count": 0,
            "alarm_hit_rate": 0.0,
            "page_hit_rate": 0.0,
            "keyword_hit_rate": 0.0,
            "overall_hit_rate": 0.0,
            "schema_ok_rate": 0.0,
            "avg_top1_score": 0.0,
        }

    top1_scores = [case["topk"][0]["score"] for case in cases if case.get("topk")]
    return {
        "query_count": total,
        "alarm_hit_rate": sum(1 for case in cases if case["alarm_hit"]) / total,
        "page_hit_rate": sum(1 for case in cases if case["page_hit"]) / total,
        "keyword_hit_rate": sum(1 for case in cases if case["keyword_hit"]) / total,
        "overall_hit_rate": sum(1 for case in cases if case["overall_hit"]) / total,
        "schema_ok_rate": sum(1 for case in cases if case["schema_ok"]) / total,
        "avg_top1_score": (sum(top1_scores) / len(top1_scores)) if top1_scores else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate TextNode + Chroma + Hybrid RAG pipeline with query cases.")
    parser.add_argument("final_blocks_jsonl")
    parser.add_argument("queries_jsonl")
    parser.add_argument("--retrievers", nargs="+", choices=["bm25", "vector", "hybrid", "hybrid_rerank", "kb_bm25", "kb_hybrid", "graph_expanded", "graph_expanded_rerank"], default=["bm25", "vector", "hybrid", "hybrid_rerank"])
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--rerank-model", default=DEFAULT_RERANK_MODEL)
    parser.add_argument("--kb-blocks-jsonl", default="")
    parser.add_argument("--kb-graph-json", default="")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    cases_path = Path(args.queries_jsonl)
    if not cases_path.exists():
        raise FileNotFoundError(f"Not found: {cases_path}")

    use_block_retrievers = any(retriever in {"bm25", "vector", "hybrid", "hybrid_rerank"} for retriever in args.retrievers)
    use_kb_retrievers = any(retriever in {"kb_bm25", "kb_hybrid", "graph_expanded", "graph_expanded_rerank"} for retriever in args.retrievers)

    pipeline = load_pipeline(
        args.final_blocks_jsonl,
        embedding_model=args.embedding_model,
        rerank_model=args.rerank_model,
    ) if use_block_retrievers else None
    node_validation = pipeline.validate_nodes() if pipeline is not None else None

    kb_retriever = None
    if use_kb_retrievers:
        doc_id = Path(args.final_blocks_jsonl).resolve().parents[0].name
        kb_blocks_path = Path(args.kb_blocks_jsonl) if args.kb_blocks_jsonl else DATA_DIR / "kb" / doc_id / "kb_blocks.jsonl"
        kb_graph_path = Path(args.kb_graph_json) if args.kb_graph_json else DATA_DIR / "kb" / doc_id / "kb_graph.json"
        kb_retriever = load_kb_graph_retriever(
            kb_blocks_path,
            kb_graph_path,
            embedding_model=args.embedding_model,
            rerank_model=args.rerank_model,
        )
    queries = read_jsonl(cases_path)

    retriever_reports: Dict[str, Any] = {}
    for retriever in args.retrievers:
        active = kb_retriever if retriever in {"kb_bm25", "kb_hybrid", "graph_expanded", "graph_expanded_rerank"} else pipeline
        if active is None:
            raise RuntimeError(f"Retriever {retriever} requested but no active pipeline is available.")
        case_results = [evaluate_case(active, case, retriever) for case in queries]
        retriever_reports[retriever] = {
            "summary": summarize_cases(case_results),
            "cases": case_results,
        }

    output = {
        "doc_id": pipeline.doc_id,
        "final_blocks_jsonl": str(Path(args.final_blocks_jsonl).resolve()),
        "queries_jsonl": str(cases_path.resolve()),
        "node_validation": node_validation,
        "embedding_validation": pipeline.embedding_validation() if pipeline is not None and any(retriever in {"vector", "hybrid", "hybrid_rerank"} for retriever in args.retrievers) else None,
        "retrievers": retriever_reports,
    }

    out_path = Path(args.out) if args.out else ensure_dir(Path(args.final_blocks_jsonl).resolve().parents[2] / "eval" / pipeline.doc_id) / "rag_pipeline_validation.json"
    write_json(out_path, output)

    print(f"[OK] doc_id: {pipeline.doc_id}")
    print(f"[OK] output: {out_path}")
    if node_validation is not None:
        print(f"[OK] text_nodes: {node_validation['text_node_count']}")
    for retriever, report in retriever_reports.items():
        summary = report["summary"]
        print("-" * 80)
        print(f"retriever: {retriever}")
        print(f"query_count: {summary['query_count']}")
        print(f"alarm_hit_rate: {summary['alarm_hit_rate']:.3f}")
        print(f"page_hit_rate: {summary['page_hit_rate']:.3f}")
        print(f"keyword_hit_rate: {summary['keyword_hit_rate']:.3f}")
        print(f"overall_hit_rate: {summary['overall_hit_rate']:.3f}")
        print(f"schema_ok_rate: {summary['schema_ok_rate']:.3f}")
        print(f"avg_top1_score: {summary['avg_top1_score']:.4f}")


if __name__ == "__main__":
    main()
