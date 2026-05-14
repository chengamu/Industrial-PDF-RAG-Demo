import argparse
import json

from llamaindex_hybrid_pipeline import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_RERANK_MODEL,
    build_metadata_filters,
    load_pipeline,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Search final_blocks with TextNode + Chroma + BM25/Vector/Hybrid retrieval.")
    parser.add_argument("final_blocks_jsonl")
    parser.add_argument("query")
    parser.add_argument("--retriever", choices=["bm25", "vector", "hybrid", "hybrid_rerank"], default="hybrid_rerank")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--before", type=int, default=2)
    parser.add_argument("--after", type=int, default=3)
    parser.add_argument("--filter", action="append", default=[], help="Metadata filter in key=value form.")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--rerank-model", default=DEFAULT_RERANK_MODEL)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    filters = build_metadata_filters(args.filter)
    pipeline = load_pipeline(
        args.final_blocks_jsonl,
        embedding_model=args.embedding_model,
        rerank_model=args.rerank_model,
    )
    node_stats = pipeline.validate_nodes()
    results = pipeline.search(
        args.query,
        retriever=args.retriever,
        top_k=args.top_k,
        before=args.before,
        after=args.after,
        filters=filters,
    )
    payload = {
        "doc_id": pipeline.doc_id,
        "query": args.query,
        "retriever": args.retriever,
        "top_k": args.top_k,
        "filters": filters,
        "node_stats": node_stats,
        "results": results,
    }

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[OK] output: {args.out}")

    print(f"[OK] doc_id: {pipeline.doc_id}")
    print(f"[OK] retriever: {args.retriever}")
    print(f"[OK] results: {len(results)}")
    for index, item in enumerate(results, start=1):
        hit = item["hit_block"]
        print("-" * 100)
        print(f"rank: {index}")
        print(f"score: {item['score']:.4f}")
        print(f"block_id: {hit.get('block_id')}")
        print(f"page_no: {hit.get('page_no')}")
        print(f"block_type: {hit.get('block_type')}")
        print(f"alarm_codes: {hit.get('alarm_codes')}")
        print(f"section_path: {hit.get('section_path')}")
        print(f"page_image: {item.get('page_image')}")
        print(f"text: {str(hit.get('raw_text') or '')[:400]}")
        print(f"score_detail: {item.get('score_detail')}")


if __name__ == "__main__":
    main()
