import argparse
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from common import DATA_DIR, canonicalize_text, extract_alarm_codes, read_jsonl, tokenize, utc_now_iso, write_jsonl


PARAM_PATTERN = re.compile(r"\b[A-Z]{2,12}(?:\s*\(\s*NO\.\s*\d+(?:#\d+)?\s*\))?(?:\s*#\d+)?(?:\s*=\s*[\dA-Z]+)?\b", re.IGNORECASE)
NUMBERED_STEP_RE = re.compile(r"^(?:STEP\s*\d+|\d+[\.\)、]|[一二三四五六七八九十]+[、.])\s*", re.IGNORECASE)
COMPONENT_KEYWORDS = [
    "主轴",
    "伺服",
    "CNC",
    "PMC",
    "PLC",
    "FSSB",
    "APC",
    "参数",
    "电缆",
    "电机",
    "编码器",
    "放大器",
    "电源",
    "刀库",
    "润滑",
    "冷却",
]


def split_paragraphs(text: str) -> List[str]:
    chunks = [part.strip() for part in re.split(r"\n{2,}", text or "") if part.strip()]
    if chunks:
        return merge_short_chunks(chunks)

    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    if lines:
        return merge_short_chunks(lines)
    return []


def merge_short_chunks(chunks: List[str], min_chars: int = 80) -> List[str]:
    merged: List[str] = []
    current = ""
    for chunk in chunks:
        if not current:
            current = chunk
            continue
        if len(current) < min_chars and len(current) + len(chunk) + 1 <= 300:
            current = f"{current}\n{chunk}"
        else:
            merged.append(current)
            current = chunk
    if current:
        merged.append(current)
    return merged


def split_procedure_steps(text: str) -> List[str]:
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    steps: List[str] = []
    current = ""
    for line in lines:
        if NUMBERED_STEP_RE.match(line):
            if current:
                steps.append(current.strip())
            current = line
        else:
            current = f"{current}\n{line}".strip() if current else line
    if current:
        steps.append(current.strip())
    return steps or split_paragraphs(text)


def split_table_rows(text: str) -> List[str]:
    rows = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    return rows or split_paragraphs(text)


def extract_parameters(text: str) -> List[str]:
    parameters = []
    for match in PARAM_PATTERN.finditer(canonicalize_text(text)):
        token = re.sub(r"\s+", " ", match.group(0)).strip(" ,.;:")
        if len(token) >= 3 and any(ch.isdigit() for ch in token):
            parameters.append(token)
    return sorted(set(parameters))


def extract_components(text: str) -> List[str]:
    source = str(text or "")
    found = [keyword for keyword in COMPONENT_KEYWORDS if keyword.lower() in source.lower()]
    return sorted(set(found))


def extract_entities(block: Dict[str, Any], text: str) -> List[str]:
    entities = []
    entities.extend(block.get("alarm_codes") or [])
    entities.extend(extract_parameters(text))
    entities.extend(extract_components(text))
    return sorted(set(entity for entity in entities if str(entity).strip()))


def infer_actions(block: Dict[str, Any], text: str, kb_type: str) -> List[str]:
    title = str(block.get("title") or "").strip()
    action_candidates: List[str] = []
    if title and title not in (block.get("alarm_codes") or []):
        action_candidates.append(title)

    first_line = str(text or "").strip().splitlines()[0] if str(text or "").strip() else ""
    if first_line and first_line != title:
        action_candidates.append(first_line[:120])

    if kb_type == "procedure" and not action_candidates:
        action_candidates.append("procedure_step")
    if kb_type == "table" and not action_candidates:
        action_candidates.append("table_row")
    if kb_type == "alarm" and not action_candidates:
        action_candidates.append("alarm_event")
    return sorted(set(candidate for candidate in action_candidates if candidate))


def split_block_to_kbs(block: Dict[str, Any]) -> List[Dict[str, Any]]:
    if block.get("block_type") == "page_context":
        return []

    block_type = str(block.get("block_type") or "text")
    raw_text = str(block.get("raw_text") or "").strip()
    if not raw_text and block_type != "image":
        return []

    if block.get("alarm_codes"):
        segments = []
        for index, alarm_code in enumerate(block.get("alarm_codes") or [], start=1):
            segments.append(
                {
                    "suffix": f"alarm_{index:02d}",
                    "kb_type": "alarm",
                    "text": raw_text or str(block.get("title") or alarm_code),
                    "entities": [alarm_code],
                    "extra": {},
                }
            )
    elif block_type == "procedure":
        segments = [
            {
                "suffix": f"step_{index:02d}",
                "kb_type": "procedure",
                "text": step,
                "entities": [],
                "extra": {"step_index": index},
            }
            for index, step in enumerate(split_procedure_steps(raw_text), start=1)
        ]
    elif block_type in {"table", "table_candidate"}:
        segments = [
            {
                "suffix": f"row_{index:02d}",
                "kb_type": "table",
                "text": row,
                "entities": [],
                "extra": {"row_index": index, "table_id": block.get("block_id")},
            }
            for index, row in enumerate(split_table_rows(raw_text), start=1)
        ]
    else:
        segments = [
            {
                "suffix": f"para_{index:02d}",
                "kb_type": "warning" if block_type == "warning" else "text",
                "text": paragraph,
                "entities": [],
                "extra": {"paragraph_index": index},
            }
            for index, paragraph in enumerate(split_paragraphs(raw_text) or ([raw_text] if raw_text else []), start=1)
        ]

    timestamp = utc_now_iso()
    kb_rows: List[Dict[str, Any]] = []
    for local_index, segment in enumerate(segments, start=1):
        text = str(segment["text"] or "").strip()
        if not text:
            continue
        entities = sorted(set(segment["entities"] + extract_entities(block, text)))
        canonical_text = canonicalize_text(text)
        kb_rows.append(
            {
                "kb_id": f"{block['block_id']}__{segment['suffix']}",
                "doc_id": block.get("doc_id"),
                "block_id": block.get("block_id"),
                "kb_type": segment["kb_type"],
                "block_type": block_type,
                "block_level": block.get("block_level"),
                "entities": entities,
                "action": infer_actions(block, text, segment["kb_type"]),
                "text": text,
                "canonical_text": canonical_text,
                "searchable_text": "\n".join(
                    part
                    for part in [
                        str(block.get("title") or "").strip(),
                        " ".join(entities),
                        str(block.get("section_path") or "").strip(),
                        canonical_text,
                    ]
                    if part
                ),
                "page_no": block.get("page_no"),
                "section_path": block.get("section_path"),
                "image_paths": list(block.get("image_paths") or []),
                "bbox": block.get("bbox"),
                "alarm_codes": list(block.get("alarm_codes") or []),
                "page_context_id": block.get("page_context_id"),
                "source": block.get("source"),
                "parse_level": block.get("parse_level"),
                "needs_refine": block.get("needs_refine"),
                "refine_reason": list(block.get("refine_reason") or []),
                "prev_block_id": block.get("prev_block_id"),
                "next_block_id": block.get("next_block_id"),
                "context_prev": list(block.get("context_prev") or []),
                "context_next": list(block.get("context_next") or []),
                "created_at": block.get("created_at") or timestamp,
                "updated_at": timestamp,
                "kb_index_in_block": local_index,
                **segment["extra"],
            }
        )
    return kb_rows


def relink_kbs(kbs: List[Dict[str, Any]]) -> None:
    block_id_to_kb_ids: Dict[str, List[str]] = {}
    for kb in kbs:
        block_id_to_kb_ids.setdefault(str(kb.get("block_id") or ""), []).append(str(kb["kb_id"]))

    for index, kb in enumerate(kbs):
        kb["prev_kb_id"] = kbs[index - 1]["kb_id"] if index > 0 else None
        kb["next_kb_id"] = kbs[index + 1]["kb_id"] if index + 1 < len(kbs) else None
        kb["context_prev_kb"] = [
            kbs[i]["kb_id"]
            for i in range(max(0, index - 2), index)
        ]
        kb["context_next_kb"] = [
            kbs[i]["kb_id"]
            for i in range(index + 1, min(len(kbs), index + 3))
        ]

        prev_block_refs: List[str] = []
        for block_id in kb.get("context_prev") or []:
            prev_block_refs.extend(block_id_to_kb_ids.get(str(block_id), [])[:2])
        kb["context_prev_kb"].extend(ref for ref in prev_block_refs if ref not in kb["context_prev_kb"])

        next_block_refs: List[str] = []
        for block_id in kb.get("context_next") or []:
            next_block_refs.extend(block_id_to_kb_ids.get(str(block_id), [])[:2])
        kb["context_next_kb"].extend(ref for ref in next_block_refs if ref not in kb["context_next_kb"])


def build_kb_blocks(final_blocks_path: str | Path) -> Tuple[List[Dict[str, Any]], Path]:
    final_blocks_file = Path(final_blocks_path)
    if not final_blocks_file.exists():
        raise FileNotFoundError(f"Not found: {final_blocks_file}")

    blocks = read_jsonl(final_blocks_file)
    kb_rows: List[Dict[str, Any]] = []
    for block in blocks:
        kb_rows.extend(split_block_to_kbs(block))

    relink_kbs(kb_rows)
    doc_id = str(blocks[0].get("doc_id") or final_blocks_file.parent.name) if blocks else final_blocks_file.parent.name
    out_path = DATA_DIR / "kb" / doc_id / "kb_blocks.jsonl"
    write_jsonl(out_path, kb_rows)
    return kb_rows, out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build knowledge blocks from final_blocks.jsonl.")
    parser.add_argument("final_blocks_jsonl")
    args = parser.parse_args()

    kb_rows, out_path = build_kb_blocks(args.final_blocks_jsonl)
    print(f"[OK] kb blocks: {len(kb_rows)}")
    print(f"[OK] output: {out_path}")
    for row in kb_rows[:5]:
        print("-" * 80)
        print(f"kb_id: {row['kb_id']}")
        print(f"block_id: {row['block_id']}")
        print(f"kb_type: {row['kb_type']} page={row['page_no']}")
        print(f"entities: {row['entities']}")
        print(f"action: {row['action']}")
        print(f"text: {row['text'][:160]}")


if __name__ == "__main__":
    main()
