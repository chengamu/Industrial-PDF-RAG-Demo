import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

from common import canonicalize_text, normalize_alarm_code, parse_query, read_json, read_jsonl, tokenize
from llamaindex_hybrid_pipeline import DashScopeEmbeddingClient, DashScopeReranker, DEFAULT_EMBEDDING_MODEL, DEFAULT_RERANK_MODEL, load_dotenv_if_available, require_dashscope_api_key


load_dotenv_if_available()


def kb_to_search_text(kb: Dict[str, Any]) -> str:
    parts = [
        " ".join(kb.get("entities") or []),
        " ".join(kb.get("action") or []),
        str(kb.get("section_path") or "").strip(),
        str(kb.get("searchable_text") or "").strip(),
        str(kb.get("canonical_text") or "").strip(),
        str(kb.get("text") or "").strip(),
    ]
    return "\n".join(part for part in parts if part).strip()


def normalize_score_map(score_map: Dict[str, float]) -> Dict[str, float]:
    if not score_map:
        return {}
    values = list(score_map.values())
    max_value = max(values)
    min_value = min(values)
    if abs(max_value - min_value) < 1e-9:
        return {key: 1.0 for key in score_map}
    return {key: (value - min_value) / (max_value - min_value) for key, value in score_map.items()}


def graph_edge_weight(relation: str) -> float:
    return {
        "mentions": 0.7,
        "triggers": 1.0,
        "depends_on": 0.8,
        "adjacent": 0.4,
        "same_page": 0.35,
        "related_context": 0.5,
    }.get(relation, 0.25)


class KBGraphRetriever:
    def __init__(
        self,
        kb_blocks_path: str | Path,
        kb_graph_path: str | Path,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        rerank_model: str = DEFAULT_RERANK_MODEL,
    ):
        self.kb_blocks_path = Path(kb_blocks_path)
        self.kb_graph_path = Path(kb_graph_path)
        if not self.kb_blocks_path.exists():
            raise FileNotFoundError(f"Not found: {self.kb_blocks_path}")
        if not self.kb_graph_path.exists():
            raise FileNotFoundError(f"Not found: {self.kb_graph_path}")

        self.kb_rows = read_jsonl(self.kb_blocks_path)
        self.graph = read_json(self.kb_graph_path)
        if not self.kb_rows:
            raise ValueError(f"No KB rows found: {self.kb_blocks_path}")

        self.doc_id = str(self.kb_rows[0].get("doc_id") or self.kb_blocks_path.parent.name)
        self.embedding_model = embedding_model
        self.rerank_model = rerank_model
        self.kb_by_id = {str(row["kb_id"]): row for row in self.kb_rows}
        self.search_ids = [str(row["kb_id"]) for row in self.kb_rows]
        self.search_text_by_id = {kb_id: kb_to_search_text(self.kb_by_id[kb_id]) for kb_id in self.search_ids}
        self.tokenized = [tokenize(self.search_text_by_id[kb_id]) for kb_id in self.search_ids]
        self.bm25 = BM25Okapi(self.tokenized)
        self.adj: Dict[str, List[Tuple[str, str, float]]] = defaultdict(list)
        for edge in self.graph.get("edges", []):
            source = str(edge.get("source") or "")
            target = str(edge.get("target") or "")
            relation = str(edge.get("relation") or "")
            if source in self.kb_by_id:
                self.adj[source].append((target, relation, graph_edge_weight(relation)))
            if target in self.kb_by_id:
                self.adj[target].append((source, relation, graph_edge_weight(relation)))

        self._embedding_client: DashScopeEmbeddingClient | None = None
        self._reranker: DashScopeReranker | None = None
        self._kb_embeddings: Dict[str, np.ndarray] | None = None

    def _get_embedding_client(self) -> DashScopeEmbeddingClient:
        if self._embedding_client is None:
            self._embedding_client = DashScopeEmbeddingClient(self.embedding_model, require_dashscope_api_key())
        return self._embedding_client

    def _get_reranker(self) -> DashScopeReranker:
        if self._reranker is None:
            self._reranker = DashScopeReranker(self.rerank_model, require_dashscope_api_key())
        return self._reranker

    def _ensure_embeddings(self) -> Dict[str, np.ndarray]:
        if self._kb_embeddings is not None:
            return self._kb_embeddings
        embedder = self._get_embedding_client()
        vectors = embedder.embed_documents([self.search_text_by_id[kb_id] for kb_id in self.search_ids])
        self._kb_embeddings = {
            kb_id: np.array(vector, dtype=float)
            for kb_id, vector in zip(self.search_ids, vectors)
        }
        return self._kb_embeddings

    def search_bm25(self, query: str, top_k: int) -> Dict[str, float]:
        parsed = parse_query(query)
        scores = np.array(self.bm25.get_scores(tokenize(parsed["canonical_query"])), dtype=float)
        score_map: Dict[str, float] = {}
        for index, kb_id in enumerate(self.search_ids):
            score = float(scores[index])
            if score > 0:
                score_map[kb_id] = score
        ranked_ids = sorted(score_map, key=score_map.get, reverse=True)[:top_k]
        return {kb_id: score_map[kb_id] for kb_id in ranked_ids}

    def search_vector(self, query: str, top_k: int) -> Dict[str, float]:
        embeddings = self._ensure_embeddings()
        query_vector = np.array(self._get_embedding_client().embed_query(query), dtype=float)
        query_vector = query_vector / (np.linalg.norm(query_vector) + 1e-9)
        score_map: Dict[str, float] = {}
        for kb_id, vector in embeddings.items():
            vector = vector / (np.linalg.norm(vector) + 1e-9)
            score_map[kb_id] = float(np.dot(vector, query_vector))
        ranked_ids = sorted(score_map, key=score_map.get, reverse=True)[:top_k]
        return {kb_id: score_map[kb_id] for kb_id in ranked_ids}

    def search_hybrid(self, query: str, top_k: int) -> Dict[str, Dict[str, float]]:
        candidate_k = max(top_k * 4, 20)
        parsed = parse_query(query)
        query_codes = {normalize_alarm_code(code) for code in parsed["alarm_codes"]}
        bm25_scores = self.search_bm25(query, candidate_k)
        vector_scores = self.search_vector(query, candidate_k)
        bm25_norm = normalize_score_map(bm25_scores)
        vector_norm = normalize_score_map(vector_scores)
        candidate_ids = set(bm25_scores) | set(vector_scores)
        scored: Dict[str, Dict[str, float]] = {}
        for kb_id in candidate_ids:
            kb = self.kb_by_id[kb_id]
            kb_codes = {normalize_alarm_code(code) for code in (kb.get("alarm_codes") or [])}
            entity_codes = {
                normalize_alarm_code(entity)
                for entity in (kb.get("entities") or [])
                if any(ch.isdigit() for ch in str(entity))
            }
            exact_match = 1.0 if query_codes and (kb_codes.intersection(query_codes) or entity_codes.intersection(query_codes)) else 0.0
            hybrid_score = 0.45 * bm25_norm.get(kb_id, 0.0) + 0.35 * vector_norm.get(kb_id, 0.0) + 0.20 * exact_match
            scored[kb_id] = {
                "bm25_score": bm25_scores.get(kb_id, 0.0),
                "vector_score": vector_scores.get(kb_id, 0.0),
                "bm25_norm": bm25_norm.get(kb_id, 0.0),
                "vector_norm": vector_norm.get(kb_id, 0.0),
                "exact_match": exact_match,
                "hybrid_score": hybrid_score,
            }
        ranked_ids = sorted(scored, key=lambda kb_id: scored[kb_id]["hybrid_score"], reverse=True)[:candidate_k]
        return {kb_id: scored[kb_id] for kb_id in ranked_ids}

    def expand_graph(self, seed_scores: Dict[str, Dict[str, float]], hops: int = 1, max_expand: int = 24) -> Dict[str, Dict[str, float]]:
        expanded = dict(seed_scores)
        frontier: Set[str] = set(seed_scores.keys())
        visited: Set[str] = set(frontier)
        for hop in range(hops):
            next_frontier: Set[str] = set()
            for kb_id in frontier:
                base_score = expanded[kb_id]["hybrid_score"]
                neighbors = self.adj.get(kb_id, [])
                neighbors = sorted(neighbors, key=lambda item: item[2], reverse=True)[:max_expand]
                for target, relation, edge_weight in neighbors:
                    if target not in self.kb_by_id or target in visited:
                        continue
                    expanded[target] = {
                        "graph_parent": kb_id,
                        "graph_relation": relation,
                        "graph_edge_weight": edge_weight,
                        "graph_hop": hop + 1,
                        "hybrid_score": base_score * edge_weight,
                        "bm25_score": 0.0,
                        "vector_score": 0.0,
                        "bm25_norm": 0.0,
                        "vector_norm": 0.0,
                        "exact_match": 0.0,
                    }
                    visited.add(target)
                    next_frontier.add(target)
            frontier = next_frontier
            if not frontier:
                break
        ranked_ids = sorted(expanded, key=lambda kb_id: expanded[kb_id]["hybrid_score"], reverse=True)[:max(max_expand, len(seed_scores))]
        return {kb_id: expanded[kb_id] for kb_id in ranked_ids}

    def rerank(self, query: str, candidate_scores: Dict[str, Dict[str, float]], top_k: int) -> Dict[str, Dict[str, float]]:
        if not candidate_scores:
            return {}
        candidate_ids = list(candidate_scores.keys())
        documents = [self.search_text_by_id[kb_id] for kb_id in candidate_ids]
        reranked = self._get_reranker().rerank(query, documents, top_n=min(top_k, len(documents)))
        final_scores: Dict[str, Dict[str, float]] = {}
        for row in reranked:
            kb_id = candidate_ids[row["index"]]
            final_scores[kb_id] = candidate_scores[kb_id] | {"rerank_score": float(row["relevance_score"])}
        return final_scores

    def expand_result(self, kb_id: str, score: float, score_detail: Dict[str, float], before: int = 2, after: int = 3) -> Dict[str, Any]:
        kb = self.kb_by_id[kb_id]
        index = self.search_ids.index(kb_id)
        context_before = [self.kb_by_id[self.search_ids[i]] for i in range(max(0, index - before), index)]
        context_after = [self.kb_by_id[self.search_ids[i]] for i in range(index + 1, min(len(self.search_ids), index + 1 + after))]
        graph_neighbors = []
        for target, relation, edge_weight in sorted(self.adj.get(kb_id, []), key=lambda item: item[2], reverse=True)[:6]:
            if target in self.kb_by_id:
                graph_neighbors.append(
                    {
                        "kb_id": target,
                        "relation": relation,
                        "edge_weight": edge_weight,
                        "text_preview": str(self.kb_by_id[target].get("text") or "")[:160],
                    }
                )
        return {
            "score": score,
            "score_detail": score_detail,
            "hit_block": kb,
            "context_blocks": context_before + context_after,
            "page_context": {"page_context_id": kb.get("page_context_id")},
            "page_image": (kb.get("image_paths") or [""])[0] if kb.get("image_paths") else "",
            "graph_neighbors": graph_neighbors,
        }

    def search(self, query: str, retriever: str = "graph_expanded_rerank", top_k: int = 10, graph_hops: int = 1) -> List[Dict[str, Any]]:
        if retriever == "kb_bm25":
            score_map = self.search_bm25(query, top_k)
            ranked = [(kb_id, score, {"bm25_score": score}) for kb_id, score in sorted(score_map.items(), key=lambda item: item[1], reverse=True)[:top_k]]
        elif retriever == "kb_hybrid":
            score_map = self.search_hybrid(query, top_k)
            ranked = [(kb_id, detail["hybrid_score"], detail) for kb_id, detail in list(score_map.items())[:top_k]]
        elif retriever == "graph_expanded":
            hybrid = self.search_hybrid(query, top_k)
            expanded = self.expand_graph(hybrid, hops=graph_hops)
            ranked = [(kb_id, detail["hybrid_score"], detail) for kb_id, detail in list(expanded.items())[:top_k]]
        elif retriever == "graph_expanded_rerank":
            hybrid = self.search_hybrid(query, top_k)
            expanded = self.expand_graph(hybrid, hops=graph_hops)
            reranked = self.rerank(query, expanded, top_k)
            ranked = [
                (kb_id, detail["rerank_score"], detail)
                for kb_id, detail in sorted(reranked.items(), key=lambda item: item[1]["rerank_score"], reverse=True)[:top_k]
            ]
        else:
            raise ValueError("retriever must be one of: kb_bm25, kb_hybrid, graph_expanded, graph_expanded_rerank")

        return [self.expand_result(kb_id, score, detail) for kb_id, score, detail in ranked]


def load_kb_graph_retriever(
    kb_blocks_path: str | Path,
    kb_graph_path: str | Path,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    rerank_model: str = DEFAULT_RERANK_MODEL,
) -> KBGraphRetriever:
    return KBGraphRetriever(
        kb_blocks_path=kb_blocks_path,
        kb_graph_path=kb_graph_path,
        embedding_model=embedding_model,
        rerank_model=rerank_model,
    )
