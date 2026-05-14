import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

from common import DATA_DIR, read_jsonl, write_json


def classify_entity(entity: str) -> str:
    if not entity:
        return "unknown"
    if entity[:2].upper() in {"SP", "SV", "PS", "DS", "SR", "SW", "EX", "OH", "PC", "OT", "IO", "APC", "FS"} and any(ch.isdigit() for ch in entity):
        return "alarm_code"
    if any(ch.isdigit() for ch in entity):
        return "parameter"
    return "component"


def make_kb_node(kb: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "node_id": kb["kb_id"],
        "node_type": "kb",
        "kb_id": kb["kb_id"],
        "block_id": kb.get("block_id"),
        "doc_id": kb.get("doc_id"),
        "page_no": kb.get("page_no"),
        "section_path": kb.get("section_path"),
        "kb_type": kb.get("kb_type"),
        "text": kb.get("text"),
        "entities": kb.get("entities") or [],
        "action": kb.get("action") or [],
        "image_paths": kb.get("image_paths") or [],
    }


def make_entity_node(entity: str) -> Dict[str, Any]:
    return {
        "node_id": f"entity::{entity}",
        "node_type": "entity",
        "entity": entity,
        "entity_type": classify_entity(entity),
    }


def append_edge(edges: List[Dict[str, Any]], source: str, target: str, relation: str, **metadata: Any) -> None:
    edges.append(
        {
            "source": source,
            "target": target,
            "relation": relation,
            **metadata,
        }
    )


def build_graph(kb_blocks_path: str | Path) -> Tuple[Dict[str, Any], Path]:
    kb_blocks_file = Path(kb_blocks_path)
    if not kb_blocks_file.exists():
        raise FileNotFoundError(f"Not found: {kb_blocks_file}")

    kb_rows = read_jsonl(kb_blocks_file)
    if not kb_rows:
        raise ValueError(f"No KB rows found: {kb_blocks_file}")

    doc_id = str(kb_rows[0].get("doc_id") or kb_blocks_file.parent.name)
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    entity_nodes: Dict[str, Dict[str, Any]] = {}
    kb_by_id = {str(kb["kb_id"]): kb for kb in kb_rows}
    page_to_kbs: Dict[int, List[str]] = {}
    block_to_kbs: Dict[str, List[str]] = {}

    for kb in kb_rows:
        nodes.append(make_kb_node(kb))
        block_to_kbs.setdefault(str(kb.get("block_id") or ""), []).append(str(kb["kb_id"]))
        page_no = kb.get("page_no")
        if isinstance(page_no, int):
            page_to_kbs.setdefault(page_no, []).append(str(kb["kb_id"]))

        for entity in kb.get("entities") or []:
            entity_nodes.setdefault(entity, make_entity_node(entity))
            append_edge(edges, str(kb["kb_id"]), f"entity::{entity}", "mentions", page_no=kb.get("page_no"))
            if classify_entity(entity) == "alarm_code":
                append_edge(edges, f"entity::{entity}", str(kb["kb_id"]), "triggers", page_no=kb.get("page_no"))
            if classify_entity(entity) == "parameter":
                append_edge(edges, str(kb["kb_id"]), f"entity::{entity}", "depends_on", page_no=kb.get("page_no"))

        prev_kb_id = kb.get("prev_kb_id")
        next_kb_id = kb.get("next_kb_id")
        if prev_kb_id and prev_kb_id in kb_by_id:
            append_edge(edges, str(prev_kb_id), str(kb["kb_id"]), "adjacent")
        if next_kb_id and next_kb_id in kb_by_id:
            append_edge(edges, str(kb["kb_id"]), str(next_kb_id), "adjacent")

    nodes.extend(entity_nodes.values())

    for page_no, kb_ids in page_to_kbs.items():
        for index in range(len(kb_ids) - 1):
            append_edge(edges, kb_ids[index], kb_ids[index + 1], "same_page", page_no=page_no)

    for kb in kb_rows:
        current_kb_id = str(kb["kb_id"])
        for block_ref in kb.get("context_prev") or []:
            for target_kb_id in block_to_kbs.get(str(block_ref), [])[:2]:
                append_edge(edges, current_kb_id, target_kb_id, "related_context", direction="prev")
        for block_ref in kb.get("context_next") or []:
            for target_kb_id in block_to_kbs.get(str(block_ref), [])[:2]:
                append_edge(edges, current_kb_id, target_kb_id, "related_context", direction="next")

    deduped_edges: List[Dict[str, Any]] = []
    seen = set()
    for edge in edges:
        key = (
            edge["source"],
            edge["target"],
            edge["relation"],
            edge.get("direction"),
            edge.get("page_no"),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped_edges.append(edge)

    graph = {
        "doc_id": doc_id,
        "node_count": len(nodes),
        "edge_count": len(deduped_edges),
        "nodes": nodes,
        "edges": deduped_edges,
    }
    out_path = DATA_DIR / "kb" / doc_id / "kb_graph.json"
    write_json(out_path, graph)
    return graph, out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a lightweight knowledge graph from kb_blocks.jsonl.")
    parser.add_argument("kb_blocks_jsonl")
    args = parser.parse_args()

    graph, out_path = build_graph(args.kb_blocks_jsonl)
    print(f"[OK] nodes: {graph['node_count']}")
    print(f"[OK] edges: {graph['edge_count']}")
    print(f"[OK] output: {out_path}")


if __name__ == "__main__":
    main()
