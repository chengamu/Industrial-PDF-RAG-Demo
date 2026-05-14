import os
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

from common import (
    DATA_DIR,
    canonicalize_text,
    ensure_dir,
    file_sha256,
    load_dotenv_if_available,
    normalize_alarm_code,
    parse_query,
    read_json,
    read_jsonl,
    tokenize,
    write_json,
)


load_dotenv_if_available()

DEFAULT_EMBEDDING_MODEL = os.getenv("DASHSCOPE_EMBEDDING_MODEL") or os.getenv("DASHSCOPE_MODEL") or "text-embedding-v4"
DEFAULT_RERANK_MODEL = os.getenv("DASHSCOPE_RERANK_MODEL", "qwen3-rerank")
DEFAULT_COLLECTION_PREFIX = "final_blocks"


def require_module(module_name: str, install_hint: str) -> Any:
    try:
        return __import__(module_name, fromlist=["*"])
    except ImportError as exc:
        raise RuntimeError(f"Missing dependency: {module_name}. {install_hint}") from exc


def require_dashscope_api_key() -> str:
    load_dotenv_if_available()
    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing DASHSCOPE_API_KEY. Export it before running vector or rerank retrieval.")
    return api_key


def build_metadata_filters(pairs: Iterable[str]) -> Dict[str, Any]:
    filters: Dict[str, Any] = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep:
            raise ValueError(f"Invalid filter '{pair}'. Use key=value.")
        key = key.strip()
        raw_value = value.strip()
        if not key:
            raise ValueError(f"Invalid filter '{pair}'. Empty key.")
        if raw_value.isdigit():
            filters[key] = int(raw_value)
        elif raw_value.lower() in {"true", "false"}:
            filters[key] = raw_value.lower() == "true"
        else:
            filters[key] = raw_value
    return filters


def block_to_text(block: Dict[str, Any]) -> str:
    parts = [
        str(block.get("title") or "").strip(),
        " ".join(block.get("alarm_codes") or []),
        str(block.get("section_path") or "").strip(),
        str(block.get("searchable_text") or "").strip(),
        str(block.get("raw_text") or "").strip(),
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


def metadata_matches(block: Dict[str, Any], filters: Dict[str, Any]) -> bool:
    for key, expected in filters.items():
        value = block.get(key)
        if isinstance(value, list):
            if expected not in value:
                return False
            continue
        if value != expected:
            return False
    return True


def chroma_where(filters: Dict[str, Any]) -> Dict[str, Any] | None:
    if not filters:
        return None
    if len(filters) == 1:
        key, value = next(iter(filters.items()))
        return {key: value}
    return {"$and": [{key: value} for key, value in filters.items()]}


def result_preview(block: Dict[str, Any], limit: int = 240) -> str:
    return str(block.get("raw_text") or "")[:limit]


class DashScopeEmbeddingClient:
    def __init__(self, model_name: str, api_key: str):
        embeddings_module = require_module(
            "langchain_community.embeddings",
            "Add 'langchain-community' to dependencies and run uv sync.",
        )
        self.model_name = model_name
        self.client = embeddings_module.DashScopeEmbeddings(
            model=model_name,
            dashscope_api_key=api_key,
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        try:
            return [list(vector) for vector in self.client.embed_documents(texts)]
        except Exception:
            if len(texts) > 1:
                middle = max(1, len(texts) // 2)
                return self.embed_documents(texts[:middle]) + self.embed_documents(texts[middle:])
            vectors: List[List[float]] = []
            for text in texts:
                vector = self.client.embed_query(text)
                vectors.append(list(vector))
            return vectors

    def embed_query(self, query: str) -> List[float]:
        return list(self.client.embed_query(query))


class DashScopeReranker:
    def __init__(self, model_name: str, api_key: str):
        dashscope_module = require_module(
            "dashscope",
            "Add 'dashscope' to dependencies and run uv sync.",
        )
        self.model_name = model_name
        self.api_key = api_key
        self.dashscope = dashscope_module
        self.dashscope.api_key = api_key

    def rerank(self, query: str, documents: List[str], top_n: int) -> List[Dict[str, Any]]:
        response = self.dashscope.TextReRank.call(
            model=self.model_name,
            query=query,
            documents=documents,
            top_n=top_n,
            return_documents=False,
        )
        if response.status_code != 200:
            raise RuntimeError(f"Rerank failed: {response.message}")
        return [
            {
                "index": int(row.index),
                "relevance_score": float(row.relevance_score),
            }
            for row in response.output.results
        ]


class FinalBlocksHybridPipeline:
    def __init__(
        self,
        final_blocks_path: str | Path,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        rerank_model: str = DEFAULT_RERANK_MODEL,
    ):
        self.final_blocks_path = Path(final_blocks_path)
        if not self.final_blocks_path.exists():
            raise FileNotFoundError(f"Not found: {self.final_blocks_path}")

        self.blocks = read_jsonl(self.final_blocks_path)
        if not self.blocks:
            raise ValueError(f"No blocks found: {self.final_blocks_path}")

        self.doc_id = str(self.blocks[0].get("doc_id") or self.final_blocks_path.parent.name)
        self.embedding_model = embedding_model
        self.rerank_model = rerank_model
        self.index_dir = ensure_dir(DATA_DIR / "index" / self.doc_id / "llamaindex_chroma")
        self.meta_path = self.index_dir / "pipeline_meta.json"
        self.block_file_sha256 = file_sha256(self.final_blocks_path)

        self.search_blocks = [block for block in self.blocks if block.get("block_type") != "page_context"]
        self.page_context_by_page = {
            int(block["page_no"]): block
            for block in self.blocks
            if block.get("block_type") == "page_context" and isinstance(block.get("page_no"), int)
        }
        self.block_by_id = {
            str(block.get("block_id")): block
            for block in self.blocks
            if block.get("block_id")
        }
        self.node_text_by_id = {
            str(block.get("block_id")): block_to_text(block)
            for block in self.search_blocks
            if block.get("block_id")
        }
        self.node_metadata_by_id = {
            str(block.get("block_id")): self._node_metadata(block)
            for block in self.search_blocks
            if block.get("block_id")
        }
        self.search_block_ids = [str(block.get("block_id")) for block in self.search_blocks if block.get("block_id")]
        self.tokenized = [tokenize(self.node_text_by_id[block_id]) for block_id in self.search_block_ids]
        self.bm25 = BM25Okapi(self.tokenized)

        self._nodes: List[Any] | None = None
        self._embedding_client: DashScopeEmbeddingClient | None = None
        self._reranker: DashScopeReranker | None = None
        self._chroma_collection = None
        self._embedding_dim: int | None = None

    def _node_metadata(self, block: Dict[str, Any]) -> Dict[str, Any]:
        return dict(block)

    def _node_cache_path(self) -> Path:
        return self.index_dir / "text_nodes.json"

    def build_text_nodes(self) -> List[Any]:
        if self._nodes is not None:
            return self._nodes

        schema_module = require_module(
            "llama_index.core.schema",
            "Add 'llama-index-core' to dependencies and run uv sync.",
        )
        text_node_class = schema_module.TextNode

        nodes = []
        for block in self.search_blocks:
            block_id = str(block.get("block_id") or "")
            if not block_id:
                continue
            nodes.append(
                text_node_class(
                    id_=block_id,
                    text=self.node_text_by_id[block_id],
                    metadata=self.node_metadata_by_id[block_id],
                )
            )

        self._nodes = nodes
        write_json(
            self._node_cache_path(),
            {
                "doc_id": self.doc_id,
                "block_file_sha256": self.block_file_sha256,
                "node_count": len(nodes),
                "nodes": [
                    {
                        "node_id": getattr(node, "node_id", self.search_block_ids[index]),
                        "text": getattr(node, "text", self.node_text_by_id[self.search_block_ids[index]]),
                        "metadata": getattr(node, "metadata", self.node_metadata_by_id[self.search_block_ids[index]]),
                    }
                    for index, node in enumerate(nodes)
                ],
            },
        )
        return nodes

    def validate_nodes(self) -> Dict[str, Any]:
        nodes = self.build_text_nodes()
        required_fields = [
            "block_id",
            "page_no",
            "section_path",
            "alarm_codes",
            "block_type",
            "page_context_id",
            "created_at",
            "updated_at",
        ]
        metadata_missing = {
            field: sum(
                1
                for node in nodes
                if field not in getattr(node, "metadata", {})
            )
            for field in required_fields
        }
        metadata_empty = {
            field: sum(
                1
                for node in nodes
                if field in {"block_id", "block_type", "page_context_id", "created_at", "updated_at"}
                and not str(getattr(node, "metadata", {}).get(field) or "").strip()
            )
            for field in ["block_id", "block_type", "page_context_id", "created_at", "updated_at"]
        }
        empty_text_count = sum(1 for node in nodes if not str(getattr(node, "text", "")).strip())
        return {
            "doc_id": self.doc_id,
            "block_file_sha256": self.block_file_sha256,
            "block_count": len(self.blocks),
            "search_block_count": len(self.search_blocks),
            "page_context_count": len(self.blocks) - len(self.search_blocks),
            "text_node_count": len(nodes),
            "empty_text_count": empty_text_count,
            "metadata_missing": metadata_missing,
            "metadata_empty": metadata_empty,
            "sample_node": {
                "node_id": getattr(nodes[0], "node_id", ""),
                "text_preview": str(getattr(nodes[0], "text", ""))[:240] if nodes else "",
                "metadata": getattr(nodes[0], "metadata", {}) if nodes else {},
            },
        }

    def _get_embedding_client(self) -> DashScopeEmbeddingClient:
        if self._embedding_client is None:
            api_key = require_dashscope_api_key()
            self._embedding_client = DashScopeEmbeddingClient(self.embedding_model, api_key)
        return self._embedding_client

    def _get_reranker(self) -> DashScopeReranker:
        if self._reranker is None:
            api_key = require_dashscope_api_key()
            self._reranker = DashScopeReranker(self.rerank_model, api_key)
        return self._reranker

    def _collection_name(self) -> str:
        doc_name = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in self.doc_id)
        return f"{DEFAULT_COLLECTION_PREFIX}_{doc_name}_{self.block_file_sha256[:12]}"

    def _expected_meta(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "block_file_sha256": self.block_file_sha256,
            "embedding_model": self.embedding_model,
            "rerank_model": self.rerank_model,
            "collection_name": self._collection_name(),
            "node_count": len(self.search_block_ids),
        }

    def ensure_vector_store(self) -> Dict[str, Any]:
        if self._chroma_collection is not None:
            return self._expected_meta() | {"embedding_dim": self._embedding_dim}

        chromadb_module = require_module(
            "chromadb",
            "Add 'chromadb' to dependencies and run uv sync.",
        )
        client = chromadb_module.PersistentClient(path=str(self.index_dir / "chroma"))
        expected_meta = self._expected_meta()
        collection = client.get_or_create_collection(
            name=expected_meta["collection_name"],
            metadata={"hnsw:space": "cosine"},
        )

        current_meta = read_json(self.meta_path) if self.meta_path.exists() else {}
        current_meta_core = {key: current_meta.get(key) for key in expected_meta}
        needs_rebuild = current_meta_core != expected_meta or collection.count() != len(self.search_block_ids)
        if needs_rebuild:
            if collection.count() > 0:
                existing = collection.get(limit=collection.count(), include=[])
                existing_ids = list(existing.get("ids") or [])
                if existing_ids:
                    collection.delete(ids=existing_ids)

            embedder = self._get_embedding_client()
            batch_size = 32
            for start in range(0, len(self.search_block_ids), batch_size):
                batch_ids = self.search_block_ids[start : start + batch_size]
                batch_texts = [self.node_text_by_id[block_id] for block_id in batch_ids]
                print(f"[EMBED] {start + 1}-{start + len(batch_ids)}/{len(self.search_block_ids)}")
                batch_embeddings = embedder.embed_documents(batch_texts)
                batch_metadatas = [self._flat_metadata(self.node_metadata_by_id[block_id]) for block_id in batch_ids]
                self._embedding_dim = len(batch_embeddings[0]) if batch_embeddings else self._embedding_dim
                collection.upsert(
                    ids=batch_ids,
                    documents=batch_texts,
                    metadatas=batch_metadatas,
                    embeddings=batch_embeddings,
                )
        else:
            query_vector = self._get_embedding_client().embed_query("health check")
            self._embedding_dim = len(query_vector)

        self._chroma_collection = collection
        meta = expected_meta | {"embedding_dim": self._embedding_dim or 0}
        write_json(self.meta_path, meta)
        return meta

    def _flat_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        flat: Dict[str, Any] = {}
        for key, value in metadata.items():
            if isinstance(value, list):
                if all(isinstance(item, (str, int, float, bool)) for item in value):
                    flat[key] = " | ".join(str(item) for item in value)
                else:
                    flat[key] = json.dumps(value, ensure_ascii=False)
            elif isinstance(value, dict):
                flat[key] = json.dumps(value, ensure_ascii=False)
            elif value is None:
                flat[key] = ""
            else:
                flat[key] = value
        return flat

    def embedding_validation(self) -> Dict[str, Any]:
        meta = self.ensure_vector_store()
        return {
            "collection_name": meta["collection_name"],
            "node_count": meta["node_count"],
            "embedding_model": meta["embedding_model"],
            "embedding_dim": meta.get("embedding_dim") or self._embedding_dim,
            "chroma_dir": str(self.index_dir / "chroma"),
        }

    def search_bm25(self, query: str, filters: Dict[str, Any], top_k: int) -> Dict[str, float]:
        parsed = parse_query(query)
        scores = np.array(self.bm25.get_scores(tokenize(parsed["canonical_query"])), dtype=float)
        score_map: Dict[str, float] = {}
        for index, block_id in enumerate(self.search_block_ids):
            block = self.block_by_id[block_id]
            if filters and not metadata_matches(block, filters):
                continue
            score = float(scores[index])
            if score > 0:
                score_map[block_id] = score
        ranked_ids = sorted(score_map, key=score_map.get, reverse=True)[:top_k]
        return {block_id: score_map[block_id] for block_id in ranked_ids}

    def search_vector(self, query: str, filters: Dict[str, Any], top_k: int) -> Dict[str, float]:
        self.ensure_vector_store()
        if self._chroma_collection is None:
            return {}

        query_embedding = self._get_embedding_client().embed_query(query)
        self._embedding_dim = len(query_embedding)
        response = self._chroma_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=chroma_where(self._flat_metadata(filters)),
            include=["distances", "metadatas", "documents"],
        )
        ids = list((response.get("ids") or [[]])[0])
        distances = list((response.get("distances") or [[]])[0])
        score_map: Dict[str, float] = {}
        for block_id, distance in zip(ids, distances):
            score_map[str(block_id)] = 1.0 / (1.0 + float(distance))
        return score_map

    def search_hybrid(self, query: str, filters: Dict[str, Any], top_k: int) -> Dict[str, Dict[str, float]]:
        candidate_k = max(top_k * 4, 20)
        parsed = parse_query(query)
        query_codes = {normalize_alarm_code(code) for code in parsed["alarm_codes"]}
        bm25_scores = self.search_bm25(query, filters, candidate_k)
        vector_scores = self.search_vector(query, filters, candidate_k)
        bm25_norm = normalize_score_map(bm25_scores)
        vector_norm = normalize_score_map(vector_scores)

        candidate_ids = set(bm25_scores) | set(vector_scores)
        scored: Dict[str, Dict[str, float]] = {}
        for block_id in candidate_ids:
            block = self.block_by_id[block_id]
            block_codes = {normalize_alarm_code(code) for code in (block.get("alarm_codes") or [])}
            exact_match = 1.0 if query_codes and block_codes.intersection(query_codes) else 0.0
            source_priority = float(block.get("source_priority") or 0.0)
            hybrid_score = (
                0.45 * bm25_norm.get(block_id, 0.0)
                + 0.35 * vector_norm.get(block_id, 0.0)
                + 0.15 * exact_match
                + 0.05 * source_priority
            )
            scored[block_id] = {
                "bm25_score": bm25_scores.get(block_id, 0.0),
                "vector_score": vector_scores.get(block_id, 0.0),
                "bm25_norm": bm25_norm.get(block_id, 0.0),
                "vector_norm": vector_norm.get(block_id, 0.0),
                "exact_match": exact_match,
                "source_priority": source_priority,
                "hybrid_score": hybrid_score,
            }
        ranked_ids = sorted(scored, key=lambda block_id: scored[block_id]["hybrid_score"], reverse=True)[:candidate_k]
        return {block_id: scored[block_id] for block_id in ranked_ids}

    def rerank(self, query: str, candidate_scores: Dict[str, Dict[str, float]], top_k: int) -> Dict[str, Dict[str, float]]:
        if not candidate_scores:
            return {}
        candidate_ids = list(candidate_scores.keys())
        documents = [self.node_text_by_id[block_id] for block_id in candidate_ids]
        reranked = self._get_reranker().rerank(query, documents, top_n=min(top_k, len(documents)))
        final_scores: Dict[str, Dict[str, float]] = {}
        for row in reranked:
            block_id = candidate_ids[row["index"]]
            final_scores[block_id] = candidate_scores[block_id] | {
                "rerank_score": float(row["relevance_score"]),
            }
        return final_scores

    def expand_result(
        self,
        block_id: str,
        score: float,
        score_detail: Dict[str, float],
        before: int,
        after: int,
    ) -> Dict[str, Any]:
        block = self.block_by_id[block_id]

        context_before: List[Dict[str, Any]] = []
        prev_block_id = block.get("prev_block_id")
        while prev_block_id and len(context_before) < before:
            prev_block = self.block_by_id.get(str(prev_block_id))
            if prev_block is None:
                break
            if prev_block.get("block_type") != "page_context":
                context_before.append(prev_block)
            prev_block_id = prev_block.get("prev_block_id")

        context_after: List[Dict[str, Any]] = []
        next_block_id = block.get("next_block_id")
        while next_block_id and len(context_after) < after:
            next_block = self.block_by_id.get(str(next_block_id))
            if next_block is None:
                break
            if next_block.get("block_type") != "page_context":
                context_after.append(next_block)
            next_block_id = next_block.get("next_block_id")

        page_no = block.get("page_no")
        page_context = self.page_context_by_page.get(page_no) if isinstance(page_no, int) else None
        page_image = ""
        for candidate in [block, page_context or {}]:
            image_paths = candidate.get("image_paths") or []
            if image_paths:
                page_image = str(image_paths[0])
                break

        return {
            "score": score,
            "score_detail": score_detail,
            "hit_block": block,
            "context_blocks": list(reversed(context_before)) + context_after,
            "page_context": page_context,
            "page_image": page_image,
        }

    def search(
        self,
        query: str,
        retriever: str = "hybrid_rerank",
        top_k: int = 10,
        before: int = 2,
        after: int = 3,
        filters: Dict[str, Any] | None = None,
    ) -> List[Dict[str, Any]]:
        filters = filters or {}
        if retriever == "bm25":
            score_map = self.search_bm25(query, filters, top_k)
            ranked = [
                (
                    block_id,
                    score,
                    {"bm25_score": score},
                )
                for block_id, score in sorted(score_map.items(), key=lambda item: item[1], reverse=True)[:top_k]
            ]
        elif retriever == "vector":
            score_map = self.search_vector(query, filters, top_k)
            ranked = [
                (
                    block_id,
                    score,
                    {"vector_score": score},
                )
                for block_id, score in sorted(score_map.items(), key=lambda item: item[1], reverse=True)[:top_k]
            ]
        elif retriever == "hybrid":
            score_map = self.search_hybrid(query, filters, top_k)
            ranked = [
                (
                    block_id,
                    detail["hybrid_score"],
                    detail,
                )
                for block_id, detail in list(score_map.items())[:top_k]
            ]
        elif retriever == "hybrid_rerank":
            hybrid_scores = self.search_hybrid(query, filters, max(top_k * 4, 20))
            reranked = self.rerank(query, hybrid_scores, top_k)
            ranked = [
                (
                    block_id,
                    detail["rerank_score"],
                    detail,
                )
                for block_id, detail in sorted(
                    reranked.items(),
                    key=lambda item: item[1]["rerank_score"],
                    reverse=True,
                )[:top_k]
            ]
        else:
            raise ValueError("retriever must be one of: bm25, vector, hybrid, hybrid_rerank")

        return [
            self.expand_result(block_id, score, score_detail, before=before, after=after)
            for block_id, score, score_detail in ranked
        ]


def load_pipeline(
    final_blocks_path: str | Path,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    rerank_model: str = DEFAULT_RERANK_MODEL,
) -> FinalBlocksHybridPipeline:
    return FinalBlocksHybridPipeline(
        final_blocks_path=final_blocks_path,
        embedding_model=embedding_model,
        rerank_model=rerank_model,
    )
