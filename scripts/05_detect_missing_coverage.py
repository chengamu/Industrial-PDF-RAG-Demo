import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from rank_bm25 import BM25Okapi

from common import DATA_DIR, canonicalize_text, ensure_dir, read_jsonl, tokenize, write_json


def load_pages_text(doc_id: str) -> List[Dict[str, Any]]:
    path = DATA_DIR / "parsed" / doc_id / "pages_text.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}. Run 00_extract_pdf_text_pages.py or 01_parse_pdf_light.py.")
    return read_jsonl(path)


def load_blocks(doc_id: str) -> List[Dict[str, Any]]:
    path = DATA_DIR / "blocks" / doc_id / "final_blocks.jsonl"
    if not path.exists():
        path = DATA_DIR / "blocks" / doc_id / "blocks.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")
    return read_jsonl(path)


def page_block_counts(blocks: List[Dict[str, Any]]) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for block in blocks:
        page_no = block.get("page_no")
        if isinstance(page_no, int) and page_no > 0:
            counts[page_no] = counts.get(page_no, 0) + 1
    return counts


def page_block_text(blocks: List[Dict[str, Any]]) -> Dict[int, str]:
    texts: Dict[int, List[str]] = {}
    for block in blocks:
        page_no = block.get("page_no")
        if not isinstance(page_no, int) or page_no <= 0:
            continue
        texts.setdefault(page_no, []).append(str(block.get("searchable_text") or block.get("raw_text") or ""))
    return {page_no: canonicalize_text("\n".join(parts)) for page_no, parts in texts.items()}


def query_pages_text(pages: List[Dict[str, Any]], query: str, top_k: int) -> List[Dict[str, Any]]:
    corpus = [canonicalize_text(str(row.get("text") or "")) for row in pages]
    bm25 = BM25Okapi([tokenize(text) for text in corpus])
    scores = np.array(bm25.get_scores(tokenize(query)), dtype=float)
    indexes = scores.argsort()[::-1][:top_k]
    results = []
    for index in indexes:
        score = float(scores[index])
        if score <= 0:
            continue
        row = pages[int(index)]
        results.append({"page_no": int(row["page_no"]), "score": score, "text_preview": str(row.get("text") or "")[:240]})
    return results


def detect_missing_coverage(doc_id: str, queries_path: str, top_k: int = 5) -> Dict[str, Any]:
    pages = load_pages_text(doc_id)
    blocks = load_blocks(doc_id)
    counts = page_block_counts(blocks)
    block_text_by_page = page_block_text(blocks)
    queries = read_jsonl(queries_path)
    missing = []

    for case in queries:
        query = str(case.get("query") or "")
        if not query:
            continue
        query_tokens = [token for token in tokenize(query) if token]
        page_hits = query_pages_text(pages, query, top_k)
        for hit in page_hits:
            page_no = hit["page_no"]
            block_count = counts.get(page_no, 0)
            block_text = block_text_by_page.get(page_no, "")
            missing_tokens = [token for token in query_tokens if token not in block_text]
            if block_count == 0 or (query_tokens and len(missing_tokens) == len(query_tokens)):
                missing.append(
                    {
                        "type": "pages_text_coverage_missing",
                        "query": query,
                        "page_no": page_no,
                        "page_score": hit["score"],
                        "block_count": block_count,
                        "missing_tokens": missing_tokens,
                        "suggest_refine_pages": [page_no],
                        "text_preview": hit["text_preview"],
                    }
                )

    dedup: Dict[tuple, Dict[str, Any]] = {}
    for item in missing:
        key = (item["query"], item["page_no"])
        if key not in dedup or item["page_score"] > dedup[key]["page_score"]:
            dedup[key] = item

    rows = sorted(dedup.values(), key=lambda item: (-item["page_score"], item["page_no"], item["query"]))
    report = {
        "doc_id": doc_id,
        "query_count": len(queries),
        "missing_count": len(rows),
        "items": rows,
    }
    out_path = ensure_dir(DATA_DIR / "refine" / doc_id) / "missing_coverage.json"
    write_json(out_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect pages where pages_text matches but final_blocks coverage is missing.")
    parser.add_argument("doc_id")
    parser.add_argument("queries_jsonl")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    report = detect_missing_coverage(args.doc_id, args.queries_jsonl, args.top_k)
    print(f"[OK] missing coverage: {report['missing_count']}")
    print(f"[OK] output: {DATA_DIR / 'refine' / args.doc_id / 'missing_coverage.json'}")
    for item in report["items"][:20]:
        print(f"page={item['page_no']} score={item['page_score']:.3f} query={item['query']} block_count={item['block_count']}")


if __name__ == "__main__":
    main()
