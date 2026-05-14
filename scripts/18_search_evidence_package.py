import argparse
import json

from llamaindex_hybrid_pipeline import DEFAULT_EMBEDDING_MODEL, DEFAULT_RERANK_MODEL
from online_evidence_retriever import load_online_evidence_retriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the online retrieval chain and output a TopK Evidence Package.")
    parser.add_argument("final_blocks_jsonl")
    parser.add_argument("kb_blocks_jsonl")
    parser.add_argument("kb_graph_json")
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=10)
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
    payload = retriever.retrieve(args.query, top_k=args.top_k)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[OK] output: {args.out}")

    print(f"[OK] query: {payload['query']}")
    print(f"[OK] parsed_query: {payload['parsed_query']}")
    print(f"[OK] reranked: {payload['debug']['reranked']}")
    for item in payload["topk"][: args.top_k]:
        print("-" * 100)
        print(f"rank: {item['rank']}")
        print(f"kb_id: {item['kb_id']}")
        print(f"block_id: {item['block_id']}")
        print(f"rerank_score: {item['rerank_score']:.4f}")
        print(f"page_no: {item['page_no']}")
        print(f"section_path: {item['section_path']}")
        print(f"sources: {item['sources']}")
        print(f"match_reason: {item['match_reason']}")
        print(f"text: {str(item['text'])[:240]}")


if __name__ == "__main__":
    main()
