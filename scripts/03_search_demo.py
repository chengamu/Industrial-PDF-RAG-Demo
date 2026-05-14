import argparse
from typing import Any, Dict, List, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

from common import DATA_DIR, ensure_dir, parse_query, read_json, read_jsonl, tokenize, write_json


DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
EXACT_MATCH_BOOST = 10.0


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return np.dot(a, b.T)


class LocalHybridSearch:
    def __init__(self, doc_id: str, blocks: List[Dict], model_name: str = DEFAULT_EMBEDDING_MODEL):
        self.doc_id = doc_id
        self.blocks = blocks
        self.search_blocks = [block for block in blocks if block.get("block_type") != "page_context"]
        self.model_name = model_name
        self.index_dir = ensure_dir(DATA_DIR / "index" / doc_id)
        self.contents = [self.make_search_text(block) for block in self.search_blocks]
        self.tokenized = [tokenize(text) for text in self.contents]
        self.bm25 = BM25Okapi(self.tokenized)
        self.embedder = None
        self.embeddings = None
        self.block_by_id = {
            str(block.get("block_id")): block
            for block in blocks
            if block.get("block_id")
        }
        self.page_context_by_page = {
            int(block["page_no"]): block
            for block in blocks
            if block.get("block_type") == "page_context" and isinstance(block.get("page_no"), int)
        }

    @staticmethod
    def make_search_text(block: Dict) -> str:
        return str(block.get("searchable_text") or block.get("canonical_text") or block.get("raw_text") or "")

    def _index_meta(self) -> Dict:
        return {
            "model_name": self.model_name,
            "block_ids": [block.get("block_id") for block in self.search_blocks],
            "contents": self.contents,
        }

    def _load_or_build_embeddings(self) -> np.ndarray:
        from sentence_transformers import SentenceTransformer

        if self.embedder is None:
            self.embedder = SentenceTransformer(self.model_name)

        meta_path = self.index_dir / "meta.json"
        emb_path = self.index_dir / "embeddings.npy"
        expected_meta = self._index_meta()

        if meta_path.exists() and emb_path.exists():
            try:
                current_meta = read_json(meta_path)
                if current_meta == expected_meta:
                    return np.load(emb_path)
            except Exception:
                pass

        print("[1] Build embeddings...")
        embeddings = self.embedder.encode(
            self.contents,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        np.save(emb_path, embeddings)
        write_json(meta_path, expected_meta)
        return embeddings

    def _ensure_embeddings(self) -> np.ndarray:
        if self.embeddings is None:
            self.embeddings = self._load_or_build_embeddings()
        return self.embeddings

    def search(self, query: str, top_k: int = 10, retriever: str = "bm25") -> List[Tuple[float, Dict]]:
        if retriever not in {"bm25", "hybrid"}:
            raise ValueError("retriever must be 'bm25' or 'hybrid'")

        parsed = parse_query(query)
        query_codes = set(parsed["alarm_codes"])
        canonical_query = parsed["canonical_query"]

        bm25_scores = np.array(self.bm25.get_scores(tokenize(canonical_query)), dtype=float)
        if bm25_scores.size and bm25_scores.max() > 0:
            bm25_scores = bm25_scores / (bm25_scores.max() + 1e-9)

        source_scores = np.zeros(len(self.search_blocks), dtype=float)
        ocr_scores = np.ones(len(self.search_blocks), dtype=float)
        exact_scores = np.zeros(len(self.search_blocks), dtype=float)
        for index, block in enumerate(self.search_blocks):
            block_codes = set(block.get("alarm_codes") or [])
            if query_codes and block_codes.intersection(query_codes):
                exact_scores[index] = 1.0
            source_scores[index] = float(block.get("source_priority") or 0.5)
            confidence = block.get("ocr_confidence")
            if isinstance(confidence, (int, float)):
                ocr_scores[index] = 0.95 + 0.05 * max(0.0, min(1.0, float(confidence)))

        if retriever == "hybrid":
            embeddings = self._ensure_embeddings()
            if self.embedder is None:
                raise RuntimeError("Embedding model was not initialized")
            q_emb = self.embedder.encode([canonical_query], normalize_embeddings=True)
            vec_scores = cosine_similarity(embeddings, q_emb).reshape(-1)
            vec_scores = (vec_scores + 1.0) / 2.0
            final_scores = (
                EXACT_MATCH_BOOST * exact_scores
                + 0.30 * bm25_scores
                + 0.20 * vec_scores
                + 0.08 * source_scores
                + 0.02 * ocr_scores
            )
        else:
            final_scores = (
                EXACT_MATCH_BOOST * exact_scores
                + 0.85 * bm25_scores
                + 0.10 * source_scores
                + 0.05 * ocr_scores
            )

        idxs = final_scores.argsort()[::-1][:top_k]
        return [(float(final_scores[index]), self.search_blocks[index]) for index in idxs]

    def search_expanded(
        self,
        query: str,
        top_k: int = 10,
        retriever: str = "bm25",
        before: int = 2,
        after: int = 3,
    ) -> List[Tuple[float, Dict[str, Any]]]:
        return [
            (
                score,
                expand_context(
                    block,
                    self.block_by_id,
                    self.page_context_by_page,
                    before=before,
                    after=after,
                ),
            )
            for score, block in self.search(query, top_k=top_k, retriever=retriever)
        ]


def expand_context(
    block: Dict[str, Any],
    block_by_id: Dict[str, Dict[str, Any]],
    page_context_by_page: Dict[int, Dict[str, Any]],
    before: int = 2,
    after: int = 3,
) -> Dict[str, Any]:
    context_before: List[Dict[str, Any]] = []
    prev_block_id = block.get("prev_block_id")
    while prev_block_id and len(context_before) < before:
        prev_block = block_by_id.get(str(prev_block_id))
        if prev_block is None:
            break
        if prev_block.get("block_type") != "page_context":
            context_before.append(prev_block)
        prev_block_id = prev_block.get("prev_block_id")

    context_after: List[Dict[str, Any]] = []
    next_block_id = block.get("next_block_id")
    while next_block_id and len(context_after) < after:
        next_block = block_by_id.get(str(next_block_id))
        if next_block is None:
            break
        if next_block.get("block_type") != "page_context":
            context_after.append(next_block)
        next_block_id = next_block.get("next_block_id")

    page_no = block.get("page_no")
    page_context = page_context_by_page.get(page_no) if isinstance(page_no, int) else None
    page_image = ""
    for candidate in [block, page_context or {}]:
        image_paths = candidate.get("image_paths") or []
        if image_paths:
            page_image = str(image_paths[0])
            break

    return {
        "hit_block": block,
        "context_blocks": list(reversed(context_before)) + context_after,
        "page_context": page_context,
        "page_image": page_image,
    }


def load_engine(doc_id: str) -> LocalHybridSearch:
    block_dir = DATA_DIR / "blocks" / doc_id
    block_path = block_dir / "final_blocks.jsonl"
    if not block_path.exists():
        block_path = block_dir / "blocks.jsonl"
    if not block_path.exists():
        raise FileNotFoundError(f"Not found: {block_path}")
    return LocalHybridSearch(doc_id, read_jsonl(block_path))


def print_result(rank: int, score: float, block: Dict) -> None:
    print("=" * 100)
    print(f"rank: {rank}")
    print(f"score: {score:.4f}")
    print(f"block_id: {block.get('block_id')}")
    print(f"type: {block.get('block_type')} level={block.get('block_level')} priority={block.get('source_priority')}")
    print(f"page: {block.get('page_no')}")
    print(f"section: {block.get('section_path')}")
    print(f"alarm_codes: {block.get('alarm_codes')}")
    print(f"images: {block.get('image_paths')} has_image={block.get('has_image')}")
    print(f"ocr_confidence: {block.get('ocr_confidence')}")
    print(f"parse_level: {block.get('parse_level')} needs_refine={block.get('needs_refine')} reason={block.get('refine_reason')}")
    print(f"layout_refs: {block.get('layout_refs')}")
    print("-" * 100)
    raw_text = str(block.get("raw_text") or "")
    canonical_text = str(block.get("canonical_text") or "")
    print("[raw_text]")
    print(raw_text[:900])
    print("[canonical_text]")
    print(canonical_text[:500])


def print_expanded_result(rank: int, score: float, item: Dict[str, Any]) -> None:
    print_result(rank, score, item["hit_block"])

    context_blocks = item.get("context_blocks") or []
    if context_blocks:
        print("[context_blocks]")
        for context_block in context_blocks:
            preview = str(context_block.get("raw_text") or "").replace("\n", " ")[:160]
            print(
                f"- {context_block.get('block_id')} "
                f"type={context_block.get('block_type')} "
                f"page={context_block.get('page_no')} "
                f"text={preview}"
            )

    page_context = item.get("page_context")
    if page_context:
        print("[page_context]")
        print(f"block_id: {page_context.get('block_id')}")
        print(f"page_image: {item.get('page_image')}")
        print(str(page_context.get("raw_text") or "")[:700])


def main(doc_id: str, query: str, top_k: int = 10, retriever: str = "bm25") -> None:
    engine = load_engine(doc_id)
    parsed = parse_query(query)
    print(f"query: {query}")
    print(f"parsed: {parsed}")
    print(f"retriever: {retriever}")

    results = engine.search_expanded(query, top_k=top_k, retriever=retriever)
    for rank, (score, item) in enumerate(results, start=1):
        print_expanded_result(rank, score, item)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search final_blocks with BM25 or hybrid retrieval.")
    parser.add_argument("doc_id")
    parser.add_argument("query")
    parser.add_argument("top_k", nargs="?", type=int, default=10)
    parser.add_argument("--retriever", choices=["bm25", "hybrid"], default="bm25")
    args = parser.parse_args()

    main(args.doc_id, args.query, args.top_k, args.retriever)
