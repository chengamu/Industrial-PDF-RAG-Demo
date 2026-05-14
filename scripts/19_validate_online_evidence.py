import argparse
from pathlib import Path
from typing import Any, Dict, List

from common import canonicalize_text, ensure_dir, normalize_alarm_code, read_jsonl, write_json
from online_evidence_retriever import DEFAULT_EMBEDDING_MODEL, DEFAULT_RERANK_MODEL, load_online_evidence_retriever, parse_user_query


def normalize_codes(codes: List[str]) -> List[str]:
    return sorted(set(normalize_alarm_code(str(code)) for code in codes if str(code).strip()))


def joined_item_text(item: Dict[str, Any]) -> str:
    return canonicalize_text(
        "\n".join(
            [
                str(item.get("text") or ""),
                str(item.get("section_path") or ""),
                " ".join(str(code) for code in (item.get("alarm_codes") or [])),
                " ".join(str(entity) for entity in (item.get("entities") or [])),
            ]
        )
    )


def merged_item_to_result(retriever, item: Dict[str, Any], rank: int) -> Dict[str, Any]:
    kb = retriever.kb_by_id[item["kb_id"]]
    return {
        "rank": rank,
        "kb_id": item["kb_id"],
        "block_id": kb.get("block_id"),
        "kb_type": kb.get("kb_type"),
        "page_no": kb.get("page_no"),
        "section_path": kb.get("section_path"),
        "text": kb.get("text"),
        "image_paths": kb.get("image_paths") or [],
        "sources": item.get("sources") or [],
        "source_scores": item.get("source_scores") or {},
        "match_reason": item.get("match_reason") or [],
        "entities": kb.get("entities") or [],
        "alarm_codes": kb.get("alarm_codes") or [],
        "merged_score": float(item.get("merged_score") or 0.0),
    }


def reranked_item_to_result(retriever, item: Dict[str, Any]) -> Dict[str, Any]:
    kb = retriever.kb_by_id[item["kb_id"]]
    return item | {"kb_type": kb.get("kb_type")}


def item_matches_case(item: Dict[str, Any], case: Dict[str, Any]) -> bool:
    expected_codes = set(normalize_codes(case.get("expected_alarm_codes") or []))
    item_codes = set(normalize_codes(item.get("alarm_codes") or []))
    if expected_codes and item_codes.intersection(expected_codes):
        return True

    expected_pages = {int(page) for page in (case.get("expected_pages") or [])}
    if expected_pages and int(item.get("page_no") or -1) in expected_pages:
        return True

    expected_keywords = [canonicalize_text(str(keyword)) for keyword in (case.get("expected_keywords") or []) if str(keyword).strip()]
    if expected_keywords:
        text = joined_item_text(item)
        if all(keyword in text for keyword in expected_keywords):
            return True

    return False


def item_type_matches(item: Dict[str, Any], expected_kb_types: List[str]) -> bool:
    if not expected_kb_types:
        return True
    return str(item.get("kb_type") or "") in set(expected_kb_types)


def first_match_rank(items: List[Dict[str, Any]], case: Dict[str, Any], top_n: int) -> int | None:
    for item in items[:top_n]:
        if item_matches_case(item, case) and item_type_matches(item, case.get("expected_kb_types") or []):
            return int(item["rank"])
    return None


def evaluate_case(retriever, case: Dict[str, Any]) -> Dict[str, Any]:
    query = str(case.get("query") or "")
    top_k = int(case.get("top_k") or 5)
    success_top_n = int(case.get("success_top_n") or top_k)

    parsed_query = parse_user_query(query)
    limits = retriever._query_limits(parsed_query, top_k)
    entity_hits = retriever.entity_match(parsed_query, top_k=limits["entity_top_k"])
    bm25_hits = retriever.bm25_retrieval(parsed_query, top_k=limits["bm25_top_k"])
    vector_skipped = retriever._should_skip_vector(parsed_query, entity_hits, bm25_hits)
    vector_hits = [] if vector_skipped else retriever.vector_retrieval(parsed_query, top_k=limits["vector_top_k"])
    graph_hits = retriever.graph_expansion(parsed_query, entity_hits + bm25_hits + vector_hits, top_k=limits["graph_top_k"])
    merged = retriever.merge_candidates(
        parsed_query,
        {
            "entity_match": entity_hits,
            "bm25": bm25_hits,
            "vector": vector_hits,
            "graph": graph_hits,
        },
        top_k=limits["merge_top_k"],
    )
    reranked = retriever.rerank_candidates(parsed_query, merged[: limits["rerank_pool"]], top_k=top_k)

    merge_topk = [merged_item_to_result(retriever, item, rank=index + 1) for index, item in enumerate(merged[:top_k])]
    rerank_topk = [reranked_item_to_result(retriever, item) for item in reranked]

    merge_hit_rank = first_match_rank(merge_topk, case, success_top_n)
    rerank_hit_rank = first_match_rank(rerank_topk, case, success_top_n)

    parse_expected_codes = set(normalize_codes(case.get("expected_alarm_codes") or []))
    parse_alarm_ok = True if not parse_expected_codes else set(parsed_query["alarm_codes"]).intersection(parse_expected_codes) == parse_expected_codes

    require_graph = bool(case.get("require_graph"))
    graph_hit = any("graph" in (item.get("sources") or []) for item in rerank_topk) if require_graph else None

    return {
        "query": query,
        "category": str(case.get("category") or ""),
        "success_top_n": success_top_n,
        "parse_alarm_ok": parse_alarm_ok,
        "vector_skipped": vector_skipped,
        "require_graph": require_graph,
        "graph_hit": graph_hit,
        "merge_hit_rank": merge_hit_rank,
        "rerank_hit_rank": rerank_hit_rank,
        "merge_hit": merge_hit_rank is not None,
        "rerank_hit": rerank_hit_rank is not None,
        "rerank_better_than_merge": merge_hit_rank is not None and rerank_hit_rank is not None and rerank_hit_rank < merge_hit_rank,
        "rerank_no_worse_than_merge": (
            (merge_hit_rank is None and rerank_hit_rank is not None)
            or (merge_hit_rank is not None and rerank_hit_rank is not None and rerank_hit_rank <= merge_hit_rank)
            or (merge_hit_rank is None and rerank_hit_rank is None)
        ),
        "parsed_query": parsed_query,
        "debug": {
            "entity_hits": len(entity_hits),
            "bm25_hits": len(bm25_hits),
            "vector_hits": len(vector_hits),
            "graph_hits": len(graph_hits),
            "merged_candidates": len(merged),
            "reranked": len(reranked),
        },
        "merge_topk": merge_topk,
        "topk": rerank_topk,
    }


def summarize_category(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(cases)
    if total == 0:
        return {
            "query_count": 0,
            "parse_alarm_ok_rate": 0.0,
            "merge_hit_rate": 0.0,
            "rerank_hit_rate": 0.0,
            "rerank_better_than_merge_rate": 0.0,
            "rerank_no_worse_than_merge_rate": 0.0,
            "graph_hit_rate": None,
        }

    graph_cases = [case for case in cases if case["require_graph"]]
    return {
        "query_count": total,
        "parse_alarm_ok_rate": sum(1 for case in cases if case["parse_alarm_ok"]) / total,
        "merge_hit_rate": sum(1 for case in cases if case["merge_hit"]) / total,
        "rerank_hit_rate": sum(1 for case in cases if case["rerank_hit"]) / total,
        "rerank_better_than_merge_rate": sum(1 for case in cases if case["rerank_better_than_merge"]) / total,
        "rerank_no_worse_than_merge_rate": sum(1 for case in cases if case["rerank_no_worse_than_merge"]) / total,
        "graph_hit_rate": (
            sum(1 for case in graph_cases if case["graph_hit"]) / len(graph_cases)
            if graph_cases
            else None
        ),
    }


def build_summary(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    categories = ["alarm", "semantic", "procedure", "safety"]
    by_category = {
        category: summarize_category([case for case in cases if case["category"] == category])
        for category in categories
    }
    overall = summarize_category(cases)
    return {"overall": overall, "by_category": by_category}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Query -> Parse -> Retrieval -> Merge -> Rerank -> TopK online evidence chain.")
    parser.add_argument("final_blocks_jsonl")
    parser.add_argument("kb_blocks_jsonl")
    parser.add_argument("kb_graph_json")
    parser.add_argument("queries_jsonl")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--rerank-model", default=DEFAULT_RERANK_MODEL)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    retriever = load_online_evidence_retriever(
        args.final_blocks_jsonl,
        args.kb_blocks_jsonl,
        args.kb_graph_json,
        embedding_model=args.embedding_model,
        rerank_model=args.rerank_model,
    )
    cases = read_jsonl(args.queries_jsonl)
    results = [evaluate_case(retriever, case) for case in cases]
    summary = build_summary(results)

    output = {
        "final_blocks_jsonl": str(Path(args.final_blocks_jsonl).resolve()),
        "kb_blocks_jsonl": str(Path(args.kb_blocks_jsonl).resolve()),
        "kb_graph_json": str(Path(args.kb_graph_json).resolve()),
        "queries_jsonl": str(Path(args.queries_jsonl).resolve()),
        "summary": summary,
        "cases": results,
    }

    out_path = Path(args.out) if args.out else ensure_dir(Path(args.final_blocks_jsonl).resolve().parents[2] / "eval" / retriever.block_pipeline.doc_id) / "online_evidence_validation.json"
    write_json(out_path, output)

    print(f"[OK] output: {out_path}")
    print(f"[OK] query_count: {len(results)}")
    for category, row in summary["by_category"].items():
        print("-" * 80)
        print(f"category: {category}")
        print(f"query_count: {row['query_count']}")
        print(f"merge_hit_rate: {row['merge_hit_rate']:.3f}")
        print(f"rerank_hit_rate: {row['rerank_hit_rate']:.3f}")
        print(f"rerank_better_than_merge_rate: {row['rerank_better_than_merge_rate']:.3f}")
        print(f"rerank_no_worse_than_merge_rate: {row['rerank_no_worse_than_merge_rate']:.3f}")
        if row["graph_hit_rate"] is not None:
            print(f"graph_hit_rate: {row['graph_hit_rate']:.3f}")


if __name__ == "__main__":
    main()
