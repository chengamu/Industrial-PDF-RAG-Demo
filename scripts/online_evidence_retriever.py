import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from common import canonicalize_text, normalize_alarm_code, read_jsonl, tokenize
from kb_graph_retriever import KBGraphRetriever, load_kb_graph_retriever
from llamaindex_hybrid_pipeline import DEFAULT_EMBEDDING_MODEL, DEFAULT_RERANK_MODEL, FinalBlocksHybridPipeline, load_pipeline


PARAM_CODE_PATTERN = re.compile(r"\b(?:PARAM(?:ETER)?\s*)?(?:NO\.)?\s*(\d{3,5}(?:#\d+)?)\b", re.IGNORECASE)
PLC_ADDRESS_PATTERN = re.compile(r"\b[A-Z]?\d{1,4}\.\d\b")
COMPONENT_ENTITIES = [
    "主轴",
    "伺服",
    "电池",
    "风扇",
    "编码器",
    "放大器",
    "PMC",
    "PLC",
    "CNC",
    "FSSB",
    "APC",
    "USB",
    "存储卡",
    "刀库",
    "润滑",
    "冷却",
    "电机",
    "电源",
    "电缆",
]
INTENT_RULES = [
    ("safety", ["安全", "警告", "注意", "禁止", "触电", "高压"]),
    ("procedure", ["步骤", "怎么操作", "如何更换", "如何设置", "怎么更换", "怎么设置", "方法", "如何查看", "查看", "画面"]),
    ("parameter_setting", ["参数", "设定", "设置值", "参数值"]),
    ("troubleshooting", ["怎么办", "怎么处理", "如何排查", "原因", "故障", "异常", "报错", "报警"]),
    ("explain", ["是什么", "什么意思", "解释"]),
]
SAFETY_KEYWORDS = {"安全", "警告", "注意", "禁止", "高压", "触电"}
FAULT_SECTION_KEYWORDS = ["故障", "报警", "维修"]
_RETRIEVER_CACHE: Dict[Tuple[str, str, str, str, str], "OnlineEvidenceRetriever"] = {}


def unique_keep_order(items: Iterable[str]) -> List[str]:
    seen = set()
    output = []
    for item in items:
        if item not in seen and item:
            seen.add(item)
            output.append(item)
    return output


def parse_intent(query: str) -> str:
    for intent, keywords in INTENT_RULES:
        if any(keyword in query for keyword in keywords):
            return intent
    return "troubleshooting"


def parse_query_type(intent: str, alarm_codes: List[str], param_codes: List[str], plc_addresses: List[str]) -> str:
    if alarm_codes and intent == "explain":
        return "alarm_explain"
    if alarm_codes and intent == "troubleshooting":
        return "alarm_diagnosis"
    if param_codes:
        return "parameter_lookup"
    if plc_addresses:
        return "plc_lookup"
    if intent == "procedure":
        return "procedure_lookup"
    if intent == "safety":
        return "safety_lookup"
    return "general_lookup"


def extract_entities(normalized_query: str) -> List[str]:
    return [entity for entity in COMPONENT_ENTITIES if entity.lower() in normalized_query.lower()]


def extract_keywords(normalized_query: str, alarm_codes: List[str], entities: List[str]) -> List[str]:
    tokens = tokenize(normalized_query)
    stop = {normalize_alarm_code(code) for code in alarm_codes} | set(entities)
    filtered = [token for token in tokens if token and token not in stop and len(token.strip()) > 0]
    return unique_keep_order(filtered)


def build_expanded_queries(alarm_codes: List[str], entities: List[str], keywords: List[str], raw_query: str, normalized_query: str) -> List[str]:
    expanded = [raw_query.strip(), normalized_query.strip()]
    for code in alarm_codes:
        expanded.append(code)
        if entities:
            expanded.append(f"{code} {' '.join(entities[:2])}".strip())
        if keywords:
            expanded.append(f"{code} {' '.join(keywords[:4])}".strip())
    if entities and keywords:
        expanded.append(f"{' '.join(entities[:2])} {' '.join(keywords[:4])}".strip())
    if entities:
        expanded.append(" ".join(entities[:3]))
    return unique_keep_order([query for query in expanded if query])


def build_vector_queries(
    alarm_codes: List[str],
    entities: List[str],
    keywords: List[str],
    raw_query: str,
    normalized_query: str,
) -> List[str]:
    queries = [raw_query.strip()]
    semantic_terms = unique_keep_order(entities[:2] + keywords[:3])
    if semantic_terms:
        queries.append(" ".join(semantic_terms))
    elif normalized_query.strip() and normalized_query.strip() != raw_query.strip():
        queries.append(normalized_query.strip())
    if alarm_codes and semantic_terms:
        queries.append(f"{alarm_codes[0]} {' '.join(semantic_terms)}".strip())
    return unique_keep_order([query for query in queries if query])[:2]


def parse_user_query(raw_query: str) -> Dict[str, Any]:
    normalized_query = canonicalize_text(raw_query)
    alarm_codes = unique_keep_order(re.findall(r"\b(?:SP|SV|PS|DS|SR|SW|EX|OH|PC|OT|IO|APC|FSSB)\d{2,5}\b", normalized_query, flags=re.IGNORECASE))
    alarm_codes = [normalize_alarm_code(code) for code in alarm_codes]
    param_codes = unique_keep_order(match.group(1).upper() for match in PARAM_CODE_PATTERN.finditer(normalized_query))
    plc_addresses = unique_keep_order(match.group(0).upper() for match in PLC_ADDRESS_PATTERN.finditer(normalized_query))
    entities = extract_entities(normalized_query)
    intent = parse_intent(raw_query)
    query_type = parse_query_type(intent, alarm_codes, param_codes, plc_addresses)
    keywords = extract_keywords(normalized_query, alarm_codes, entities)
    need_safety = intent == "safety" or any(keyword in raw_query for keyword in SAFETY_KEYWORDS)
    expanded_queries = build_expanded_queries(alarm_codes, entities, keywords, raw_query, normalized_query)
    vector_queries = build_vector_queries(alarm_codes, entities, keywords, raw_query, normalized_query)
    return {
        "raw_query": raw_query,
        "normalized_query": normalized_query,
        "alarm_codes": alarm_codes,
        "param_codes": param_codes,
        "plc_addresses": plc_addresses,
        "entities": entities,
        "keywords": keywords,
        "intent": intent,
        "query_type": query_type,
        "need_safety": need_safety,
        "expanded_queries": expanded_queries,
        "vector_queries": vector_queries,
    }


class OnlineEvidenceRetriever:
    def __init__(
        self,
        final_blocks_jsonl: str | Path,
        kb_blocks_jsonl: str | Path,
        kb_graph_json: str | Path,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        rerank_model: str = DEFAULT_RERANK_MODEL,
    ):
        self.block_pipeline: FinalBlocksHybridPipeline = load_pipeline(final_blocks_jsonl, embedding_model=embedding_model, rerank_model=rerank_model)
        self.kb_retriever: KBGraphRetriever = load_kb_graph_retriever(kb_blocks_jsonl, kb_graph_json, embedding_model=embedding_model, rerank_model=rerank_model)
        self.kb_rows = self.kb_retriever.kb_rows
        self.kb_by_id = self.kb_retriever.kb_by_id
        self.block_by_id = self.block_pipeline.block_by_id
        self.kb_ids_by_block_id: Dict[str, List[str]] = defaultdict(list)
        self.kb_ids_by_alarm: Dict[str, List[str]] = defaultdict(list)
        self.kb_ids_by_entity: Dict[str, List[str]] = defaultdict(list)
        self.kb_ids_by_param: Dict[str, List[str]] = defaultdict(list)
        self.kb_ids_by_plc: Dict[str, List[str]] = defaultdict(list)
        for kb in self.kb_rows:
            kb_id = str(kb["kb_id"])
            block_id = str(kb.get("block_id") or "")
            if block_id:
                self.kb_ids_by_block_id[block_id].append(kb_id)
            for code in kb.get("alarm_codes") or []:
                self.kb_ids_by_alarm[normalize_alarm_code(str(code))].append(kb_id)
            for entity in kb.get("entities") or []:
                entity_str = str(entity).strip()
                if not entity_str:
                    continue
                self.kb_ids_by_entity[entity_str].append(kb_id)
                if any(ch.isdigit() for ch in entity_str):
                    self.kb_ids_by_param[entity_str.upper()].append(kb_id)
                if PLC_ADDRESS_PATTERN.fullmatch(entity_str.upper()):
                    self.kb_ids_by_plc[entity_str.upper()].append(kb_id)

    def _candidate_template(self, kb_id: str) -> Dict[str, Any]:
        kb = self.kb_by_id[kb_id]
        return {
            "kb_id": kb_id,
            "block_id": kb.get("block_id"),
            "page_no": kb.get("page_no"),
            "section_path": kb.get("section_path"),
            "text": kb.get("text"),
            "image_paths": kb.get("image_paths") or [],
            "entities": kb.get("entities") or [],
            "alarm_codes": kb.get("alarm_codes") or [],
            "sources": [],
            "source_scores": {},
            "match_reason": [],
            "matched_entities": [],
            "metadata_boost": 0.0,
        }

    def _query_limits(self, parsed_query: Dict[str, Any], top_k: int) -> Dict[str, int]:
        has_exact_code = bool(
            parsed_query["alarm_codes"] or parsed_query["param_codes"] or parsed_query["plc_addresses"]
        )
        limits = {
            "entity_top_k": 20 if has_exact_code else 12,
            "bm25_top_k": 24 if has_exact_code else 20,
            "vector_top_k": 10 if has_exact_code else 16,
            "graph_top_k": 16 if has_exact_code else 20,
            "merge_top_k": min(max(top_k * 4, 16), 28),
            "rerank_pool": min(max(top_k * 2, 12), 18),
        }
        if parsed_query["query_type"] in {"procedure_lookup", "safety_lookup"}:
            limits["vector_top_k"] = max(limits["vector_top_k"], 14)
            limits["graph_top_k"] = max(limits["graph_top_k"], 18)
        return limits

    def _should_skip_vector(
        self,
        parsed_query: Dict[str, Any],
        entity_hits: List[Dict[str, Any]],
        bm25_hits: List[Dict[str, Any]],
    ) -> bool:
        has_exact_structured_key = bool(
            parsed_query["alarm_codes"] or parsed_query["param_codes"] or parsed_query["plc_addresses"]
        )
        if not has_exact_structured_key:
            return False
        exact_candidate_count = len(entity_hits) + len(bm25_hits)
        return exact_candidate_count > 0

    def entity_match(self, parsed_query: Dict[str, Any], top_k: int = 20) -> List[Dict[str, Any]]:
        candidates: Dict[str, Dict[str, Any]] = {}

        def add(kb_id: str, source_score: float, reason: str, entity: str) -> None:
            candidate = candidates.setdefault(kb_id, self._candidate_template(kb_id))
            candidate["sources"] = unique_keep_order(candidate["sources"] + ["entity_match"])
            candidate["source_scores"]["entity_match"] = max(source_score, candidate["source_scores"].get("entity_match", 0.0))
            candidate["match_reason"] = unique_keep_order(candidate["match_reason"] + [reason])
            candidate["matched_entities"] = unique_keep_order(candidate["matched_entities"] + [entity])

        for code in parsed_query["alarm_codes"]:
            for kb_id in self.kb_ids_by_alarm.get(normalize_alarm_code(code), []):
                add(kb_id, 1.00, "alarm_code_exact_match", code)
        for param_code in parsed_query["param_codes"]:
            for kb_id in self.kb_ids_by_param.get(param_code.upper(), []):
                add(kb_id, 0.95, "param_code_exact_match", param_code)
        for plc_address in parsed_query["plc_addresses"]:
            for kb_id in self.kb_ids_by_plc.get(plc_address.upper(), []):
                add(kb_id, 0.95, "plc_address_exact_match", plc_address)
        for entity in parsed_query["entities"]:
            for kb_id in self.kb_ids_by_entity.get(entity, []):
                add(kb_id, 0.75, "entity_exact_match", entity)

        ranked = sorted(candidates.values(), key=lambda item: item["source_scores"].get("entity_match", 0.0), reverse=True)[:top_k]
        return ranked

    def _map_block_results_to_kbs(self, results: List[Dict[str, Any]], source: str, query_label: str) -> List[Dict[str, Any]]:
        mapped: List[Dict[str, Any]] = []
        for item in results:
            block = item["hit_block"]
            block_id = str(block.get("block_id") or "")
            score = float(item.get("score") or 0.0)
            reasons = []
            if source == "bm25":
                reasons.append("keyword_match")
            elif source == "vector":
                reasons.append("semantic_match")
            for kb_id in self.kb_ids_by_block_id.get(block_id, [])[:3]:
                candidate = self._candidate_template(kb_id)
                candidate["sources"] = [source]
                candidate["source_scores"] = {source: score}
                candidate["match_reason"] = reasons
                candidate["matched_query"] = query_label
                mapped.append(candidate)
        return mapped

    def bm25_retrieval(self, parsed_query: Dict[str, Any], top_k: int = 30) -> List[Dict[str, Any]]:
        candidates: Dict[str, Dict[str, Any]] = {}
        for expanded_query in parsed_query["expanded_queries"]:
            results = self.block_pipeline.search(expanded_query, retriever="bm25", top_k=top_k)
            for candidate in self._map_block_results_to_kbs(results, "bm25", expanded_query):
                kb_id = candidate["kb_id"]
                current = candidates.setdefault(kb_id, self._candidate_template(kb_id))
                current["sources"] = unique_keep_order(current["sources"] + ["bm25"])
                current["source_scores"]["bm25"] = max(candidate["source_scores"]["bm25"], current["source_scores"].get("bm25", 0.0))
                current["match_reason"] = unique_keep_order(current["match_reason"] + candidate["match_reason"])
        ranked = sorted(candidates.values(), key=lambda item: item["source_scores"].get("bm25", 0.0), reverse=True)[:top_k]
        return ranked

    def vector_retrieval(self, parsed_query: Dict[str, Any], top_k: int = 30) -> List[Dict[str, Any]]:
        candidates: Dict[str, Dict[str, Any]] = {}
        for vector_query in parsed_query["vector_queries"]:
            results = self.block_pipeline.search(vector_query, retriever="vector", top_k=top_k)
            for candidate in self._map_block_results_to_kbs(results, "vector", vector_query):
                kb_id = candidate["kb_id"]
                current = candidates.setdefault(kb_id, self._candidate_template(kb_id))
                current["sources"] = unique_keep_order(current["sources"] + ["vector"])
                current["source_scores"]["vector"] = max(candidate["source_scores"]["vector"], current["source_scores"].get("vector", 0.0))
                current["match_reason"] = unique_keep_order(current["match_reason"] + candidate["match_reason"])
        ranked = sorted(candidates.values(), key=lambda item: item["source_scores"].get("vector", 0.0), reverse=True)[:top_k]
        return ranked

    def graph_expansion(self, parsed_query: Dict[str, Any], seed_candidates: List[Dict[str, Any]], top_k: int = 30) -> List[Dict[str, Any]]:
        seed_scores: Dict[str, Dict[str, float]] = {}
        for candidate in seed_candidates[:20]:
            kb_id = candidate["kb_id"]
            base_score = max(candidate["source_scores"].values()) if candidate["source_scores"] else 0.5
            seed_scores[kb_id] = {
                "hybrid_score": base_score,
                "bm25_score": candidate["source_scores"].get("bm25", 0.0),
                "vector_score": candidate["source_scores"].get("vector", 0.0),
                "exact_match": candidate["source_scores"].get("entity_match", 0.0),
            }
        hops = 2 if parsed_query["alarm_codes"] else 1
        expanded = self.kb_retriever.expand_graph(seed_scores, hops=hops, max_expand=top_k)
        results: List[Dict[str, Any]] = []
        for kb_id, detail in list(expanded.items())[:top_k]:
            candidate = self._candidate_template(kb_id)
            candidate["sources"] = ["graph"]
            candidate["source_scores"] = {"graph": float(detail.get("hybrid_score") or 0.0)}
            relation = str(detail.get("graph_relation") or "graph_neighbor")
            candidate["match_reason"] = ["graph_neighbor", relation]
            candidate["seed_kb_id"] = detail.get("graph_parent")
            candidate["graph_distance"] = int(detail.get("graph_hop") or 0)
            results.append(candidate)
        return results

    def merge_candidates(self, parsed_query: Dict[str, Any], groups: Dict[str, List[Dict[str, Any]]], top_k: int = 40) -> List[Dict[str, Any]]:
        merged: Dict[str, Dict[str, Any]] = {}
        for candidates in groups.values():
            for candidate in candidates:
                kb_id = candidate["kb_id"]
                row = merged.setdefault(kb_id, self._candidate_template(kb_id))
                row["sources"] = unique_keep_order(row["sources"] + candidate.get("sources", []))
                row["match_reason"] = unique_keep_order(row["match_reason"] + candidate.get("match_reason", []))
                row["matched_entities"] = unique_keep_order(row.get("matched_entities", []) + candidate.get("matched_entities", []))
                for source, score in candidate.get("source_scores", {}).items():
                    row["source_scores"][source] = max(float(score), row["source_scores"].get(source, 0.0))
                if candidate.get("seed_kb_id"):
                    row["seed_kb_id"] = candidate["seed_kb_id"]
                if candidate.get("graph_distance") is not None:
                    current_distance = row.get("graph_distance")
                    row["graph_distance"] = candidate["graph_distance"] if current_distance is None else min(current_distance, candidate["graph_distance"])

        for row in merged.values():
            boost = 0.0
            kb = self.kb_by_id[row["kb_id"]]
            codes = {normalize_alarm_code(code) for code in (kb.get("alarm_codes") or [])}
            entities = set(str(entity) for entity in (kb.get("entities") or []))
            section_path = str(kb.get("section_path") or "")
            kb_type = str(kb.get("kb_type") or "")
            if parsed_query["alarm_codes"] and codes.intersection({normalize_alarm_code(code) for code in parsed_query["alarm_codes"]}):
                boost += 0.30
            if kb_type in {"alarm", "procedure"}:
                boost += 0.20
            if any(keyword in section_path for keyword in FAULT_SECTION_KEYWORDS):
                boost += 0.15
            if row.get("graph_distance") == 1:
                boost += 0.10
            if parsed_query["need_safety"] and kb_type == "warning":
                boost += 0.15
            row["metadata_boost"] = boost
            row["merged_score"] = (
                1.20 * row["source_scores"].get("entity_match", 0.0)
                + 1.00 * row["source_scores"].get("bm25", 0.0)
                + 0.90 * row["source_scores"].get("vector", 0.0)
                + 0.70 * row["source_scores"].get("graph", 0.0)
                + boost
            )
        ranked = sorted(merged.values(), key=lambda item: item["merged_score"], reverse=True)[:top_k]
        return ranked

    def rerank_candidates(self, parsed_query: Dict[str, Any], candidates: List[Dict[str, Any]], top_k: int = 10) -> List[Dict[str, Any]]:
        if not candidates:
            return []
        candidate_scores = {}
        for candidate in candidates:
            kb = self.kb_by_id[candidate["kb_id"]]
            candidate_scores[candidate["kb_id"]] = {
                "hybrid_score": candidate["merged_score"],
                "bm25_score": candidate["source_scores"].get("bm25", 0.0),
                "vector_score": candidate["source_scores"].get("vector", 0.0),
                "entity_score": candidate["source_scores"].get("entity_match", 0.0),
                "graph_score": candidate["source_scores"].get("graph", 0.0),
                "metadata_boost": candidate["metadata_boost"],
            }
        reranked = self.kb_retriever.rerank(parsed_query["raw_query"], candidate_scores, top_k=len(candidate_scores))
        ranked_items = []
        adjusted_items = []
        query_codes = {normalize_alarm_code(code) for code in parsed_query["alarm_codes"]}
        normalized_query = canonicalize_text(parsed_query["normalized_query"])
        for kb_id, detail in reranked.items():
            kb = self.kb_by_id[kb_id]
            merged = next(candidate for candidate in candidates if candidate["kb_id"] == kb_id)
            kb_type = str(kb.get("kb_type") or "")
            kb_text = canonicalize_text(
                "\n".join(
                    [
                        str(kb.get("text") or ""),
                        str(kb.get("section_path") or ""),
                        " ".join(kb.get("alarm_codes") or []),
                        " ".join(str(entity) for entity in (kb.get("entities") or [])),
                    ]
                )
            )
            post_boost = 0.0
            if normalized_query and normalized_query in kb_text:
                post_boost += 0.25
            if parsed_query["query_type"] == "procedure_lookup" and kb_type == "procedure":
                post_boost += 0.20
            if parsed_query["query_type"] == "safety_lookup" and kb_type == "warning":
                post_boost += 0.18
            kb_codes = {normalize_alarm_code(code) for code in (kb.get("alarm_codes") or [])}
            if parsed_query["query_type"] in {"alarm_diagnosis", "alarm_explain"} and query_codes.intersection(kb_codes):
                post_boost += 0.20
                if kb_type == "alarm":
                    post_boost += 0.10
            adjusted_score = float(detail["rerank_score"]) + post_boost
            adjusted_items.append(
                {
                    "kb_id": kb_id,
                    "block_id": kb.get("block_id"),
                    "rerank_score": float(detail["rerank_score"]),
                    "adjusted_rerank_score": adjusted_score,
                    "post_rerank_boost": post_boost,
                    "merged_score": float(merged["merged_score"]),
                    "page_no": kb.get("page_no"),
                    "section_path": kb.get("section_path"),
                    "text": kb.get("text"),
                    "image_paths": kb.get("image_paths") or [],
                    "sources": merged["sources"],
                    "source_scores": merged["source_scores"],
                    "match_reason": merged["match_reason"],
                    "entities": kb.get("entities") or [],
                    "alarm_codes": kb.get("alarm_codes") or [],
                }
            )
        for rank, item in enumerate(sorted(adjusted_items, key=lambda row: row["adjusted_rerank_score"], reverse=True)[:top_k], start=1):
            ranked_items.append(item | {"rank": rank})
        return ranked_items

    def retrieve(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        parsed_query = parse_user_query(query)
        limits = self._query_limits(parsed_query, top_k)
        entity_hits = self.entity_match(parsed_query, top_k=limits["entity_top_k"])
        bm25_hits = self.bm25_retrieval(parsed_query, top_k=limits["bm25_top_k"])
        vector_skipped = self._should_skip_vector(parsed_query, entity_hits, bm25_hits)
        vector_hits = [] if vector_skipped else self.vector_retrieval(parsed_query, top_k=limits["vector_top_k"])
        graph_hits = self.graph_expansion(parsed_query, entity_hits + bm25_hits + vector_hits, top_k=limits["graph_top_k"])
        merged = self.merge_candidates(
            parsed_query,
            {
                "entity_match": entity_hits,
                "bm25": bm25_hits,
                "vector": vector_hits,
                "graph": graph_hits,
            },
            top_k=limits["merge_top_k"],
        )
        topk = self.rerank_candidates(parsed_query, merged[: limits["rerank_pool"]], top_k=top_k)
        return {
            "query": query,
            "parsed_query": parsed_query,
            "topk": topk,
            "debug": {
                "entity_hits": len(entity_hits),
                "bm25_hits": len(bm25_hits),
                "vector_hits": len(vector_hits),
                "vector_skipped": vector_skipped,
                "graph_hits": len(graph_hits),
                "merged_candidates": len(merged),
                "reranked": len(topk),
            },
        }


def load_online_evidence_retriever(
    final_blocks_jsonl: str | Path,
    kb_blocks_jsonl: str | Path,
    kb_graph_json: str | Path,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    rerank_model: str = DEFAULT_RERANK_MODEL,
) -> OnlineEvidenceRetriever:
    cache_key = (
        str(Path(final_blocks_jsonl).resolve()),
        str(Path(kb_blocks_jsonl).resolve()),
        str(Path(kb_graph_json).resolve()),
        embedding_model,
        rerank_model,
    )
    if cache_key not in _RETRIEVER_CACHE:
        _RETRIEVER_CACHE[cache_key] = OnlineEvidenceRetriever(
            final_blocks_jsonl=final_blocks_jsonl,
            kb_blocks_jsonl=kb_blocks_jsonl,
            kb_graph_json=kb_graph_json,
            embedding_model=embedding_model,
            rerank_model=rerank_model,
        )
    return _RETRIEVER_CACHE[cache_key]
