import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from common import (
    DATA_DIR,
    MAX_BLOCK_CHARS,
    PROCEDURE_CHUNK_RANGE,
    SOURCE_PRIORITIES,
    TEXT_CHUNK_RANGE,
    canonicalize_text,
    ensure_dir,
    extract_alarm_codes,
    make_searchable_text,
    normalize_optional_float,
    read_json,
    utc_now_iso,
    write_jsonl,
)


ALARM_DENSE_THRESHOLD = 3
MAJOR_NUMERIC_HEADING_RE = re.compile(r"^(?P<number>\d+)\s+.+")
DECIMAL_HEADING_RE = re.compile(r"^\d+\.\d+(?:\.\d+)*\s+.+")
APPENDIX_SECTION_RE = re.compile(r"^[A-Z]\.\d+(?:\.\d+)*\s+.+")
APPENDIX_HEADING_RE = re.compile(r"^\u9644\u5f55\s*[A-Z]?\s*.+")
CHAPTER_HEADING_RE = re.compile(r"^\u7b2c\s*\d+\s*\u7ae0\s*.+")


def guess_page_no(obj: Dict[str, Any]) -> Optional[int]:
    for key in ("page_no", "page", "page_idx"):
        value = obj.get(key)
        if isinstance(value, int):
            return value + 1 if key == "page_idx" else value

    prov = obj.get("prov")
    if isinstance(prov, list):
        for item in prov:
            if isinstance(item, dict):
                page = guess_page_no(item)
                if page:
                    return page
    return None


def extract_bbox(obj: Dict[str, Any]) -> Optional[List[float]]:
    candidates = [obj.get("bbox"), obj.get("bounding_box"), obj.get("rect")]
    prov = obj.get("prov")
    if isinstance(prov, list):
        for item in prov:
            if isinstance(item, dict):
                candidates.extend([item.get("bbox"), item.get("bounding_box"), item.get("rect")])

    for value in candidates:
        if isinstance(value, list) and len(value) == 4 and all(isinstance(v, (int, float)) for v in value):
            return [float(v) for v in value]
        if isinstance(value, dict):
            keys = ("l", "t", "r", "b")
            if all(isinstance(value.get(k), (int, float)) for k in keys):
                return [float(value[k]) for k in keys]
            keys = ("x0", "y0", "x1", "y1")
            if all(isinstance(value.get(k), (int, float)) for k in keys):
                return [float(value[k]) for k in keys]
    return None


def extract_ocr_confidence(obj: Dict[str, Any]) -> Optional[float]:
    for key in ("ocr_confidence", "confidence", "confid"):
        value = normalize_optional_float(obj.get(key))
        if value is not None:
            return value

    prov = obj.get("prov")
    if isinstance(prov, list):
        values = []
        for item in prov:
            if isinstance(item, dict):
                value = extract_ocr_confidence(item)
                if value is not None:
                    values.append(value)
        if values:
            return sum(values) / len(values)
    return None


def walk_layout(obj: Any, result: List[Dict[str, Any]], path: str = "$", current_page: Optional[int] = None) -> None:
    if isinstance(obj, dict):
        page_no = guess_page_no(obj) or current_page
        label = obj.get("label") or obj.get("type") or obj.get("name")
        text = obj.get("text") or obj.get("orig") or obj.get("content")

        if isinstance(text, str) and text.strip():
            result.append(
                {
                    "layout_ref": path,
                    "layout_type": str(label or "text"),
                    "page_no": page_no,
                    "text": text.strip(),
                    "bbox": extract_bbox(obj),
                    "ocr_confidence": extract_ocr_confidence(obj),
                }
            )
        elif label and str(label).lower() in {"picture", "image", "figure"}:
            result.append(
                {
                    "layout_ref": path,
                    "layout_type": str(label),
                    "page_no": page_no,
                    "text": "",
                    "bbox": extract_bbox(obj),
                    "ocr_confidence": extract_ocr_confidence(obj),
                }
            )

        for key, value in obj.items():
            walk_layout(value, result, f"{path}.{key}", page_no)
    elif isinstance(obj, list):
        for index, item in enumerate(obj):
            walk_layout(item, result, f"{path}[{index}]", current_page)


def is_major_numeric_heading(text: str) -> bool:
    match = MAJOR_NUMERIC_HEADING_RE.match(text)
    return bool(match and int(match.group("number")) >= 10)


def is_chapter_heading(text: str) -> bool:
    return bool(
        is_major_numeric_heading(text)
        or APPENDIX_HEADING_RE.match(text)
        or CHAPTER_HEADING_RE.match(text)
    )


def is_heading(text: str) -> bool:
    text = text.strip()
    if not text or len(text) > 90:
        return False
    return bool(
        is_chapter_heading(text)
        or DECIMAL_HEADING_RE.match(text)
        or APPENDIX_SECTION_RE.match(text)
    )


def update_hierarchy(hierarchy: Dict[str, str], heading: str) -> Dict[str, str]:
    heading = heading.strip()
    next_hierarchy = dict(hierarchy)
    if is_chapter_heading(heading):
        next_hierarchy = {"chapter": heading, "section": "", "subsection": ""}
    elif DECIMAL_HEADING_RE.match(heading) or APPENDIX_SECTION_RE.match(heading):
        next_hierarchy["section"] = heading
        next_hierarchy["subsection"] = ""
    else:
        next_hierarchy["subsection"] = heading
    return next_hierarchy


def section_path(hierarchy: Dict[str, str]) -> str:
    return " > ".join(
        value
        for value in [hierarchy.get("chapter"), hierarchy.get("section"), hierarchy.get("subsection")]
        if value
    )


def classify_text(text: str, layout_type: str) -> str:
    lower_type = layout_type.lower()
    if "table" in lower_type:
        return "table_candidate"
    if re.search(r"(\u8b66\u544a|\u6ce8\u610f|\u5371\u9669|\u5c0f\u5fc3|WARNING|CAUTION|DANGER)", text, re.IGNORECASE):
        return "warning"
    if re.search(r"(^|\n)\s*(\u6b65\u9aa4|STEP|\d+[\).\u3001])\s*", text, re.IGNORECASE):
        return "procedure"
    return "text"


def chunk_text(text: str, max_chars: int) -> List[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []

    chunks: List[str] = []
    paragraphs = [part.strip() for part in re.split(r"\n{2,}", text) if part.strip()]
    current = ""
    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            if current:
                chunks.append(current.strip())
                current = ""
            for start in range(0, len(paragraph), max_chars):
                chunks.append(paragraph[start : start + max_chars].strip())
            continue
        if current and len(current) + len(paragraph) + 2 > max_chars:
            chunks.append(current.strip())
            current = paragraph
        else:
            current = f"{current}\n\n{paragraph}".strip() if current else paragraph
    if current:
        chunks.append(current.strip())
    return chunks


def make_block(
    doc_id: str,
    block_type: str,
    page_no: int,
    raw_text: str,
    hierarchy: Dict[str, str],
    layout_refs: List[str],
    image_paths: List[str],
    block_index: int,
    bbox: Optional[List[float]] = None,
    ocr_confidence: Optional[float] = None,
) -> Dict[str, Any]:
    canonical_text = canonicalize_text(raw_text)
    alarm_codes = extract_alarm_codes(canonical_text)
    title = alarm_codes[0] if alarm_codes else (
        hierarchy.get("subsection") or hierarchy.get("section") or hierarchy.get("chapter") or ""
    )
    timestamp = utc_now_iso()
    block = {
        "block_id": f"{doc_id}_b{block_index:06d}",
        "doc_id": doc_id,
        "block_type": block_type,
        "block_level": 2 if block_type in {"alarm", "alarm_candidate"} else 3,
        "source_priority": SOURCE_PRIORITIES.get(block_type, 0.5),
        "page_no": page_no,
        "title": title,
        "raw_text": raw_text,
        "canonical_text": canonical_text,
        "searchable_text": "",
        "alarm_codes": alarm_codes,
        "section_path": section_path(hierarchy),
        "hierarchy": {
            "chapter": hierarchy.get("chapter", ""),
            "section": hierarchy.get("section", ""),
            "subsection": hierarchy.get("subsection", ""),
        },
        "prev_block_id": None,
        "next_block_id": None,
        "parent_block_id": None,
        "context_prev": [],
        "context_next": [],
        "page_context_id": f"{doc_id}_page_{page_no:04d}_context" if page_no > 0 else None,
        "layout_refs": sorted(set(layout_refs)),
        "bbox": bbox,
        "image_paths": image_paths,
        "has_image": bool(image_paths),
        "ocr_confidence": ocr_confidence,
        "source": "docling_layout",
        "needs_refine": False,
        "refine_reason": [],
        "parse_level": "light",
        "created_at": timestamp,
        "updated_at": timestamp,
    }
    block["searchable_text"] = make_searchable_text(block)
    return block


def mark_refine_need(block: Dict[str, Any]) -> None:
    reasons: List[str] = []
    if block["block_type"] == "alarm_candidate":
        reasons.append("alarm_candidate")
    if block["block_type"] == "table_candidate":
        reasons.append("table_candidate")
    if block["block_type"] in {"alarm", "alarm_candidate"} and len(block.get("alarm_codes") or []) >= ALARM_DENSE_THRESHOLD:
        reasons.append("alarm_dense")
    if block["block_type"] == "image":
        reasons.append("image")

    block["needs_refine"] = bool(reasons)
    block["refine_reason"] = reasons


def apply_page_refine_flags(blocks: List[Dict[str, Any]]) -> None:
    alarm_counts: Dict[int, int] = {}
    for block in blocks:
        page_no = block.get("page_no")
        if isinstance(page_no, int) and block.get("alarm_codes"):
            alarm_counts[page_no] = alarm_counts.get(page_no, 0) + len(block.get("alarm_codes") or [])

    dense_pages = {page_no for page_no, count in alarm_counts.items() if count >= ALARM_DENSE_THRESHOLD}
    for block in blocks:
        if block.get("page_no") not in dense_pages:
            continue
        if block.get("block_type") in {"alarm_candidate", "table_candidate", "text"}:
            reasons = list(block.get("refine_reason") or [])
            if "alarm_dense" not in reasons:
                reasons.append("alarm_dense")
            block["needs_refine"] = True
            block["refine_reason"] = reasons


def split_alarm_segments(text: str) -> List[str]:
    canonical = canonicalize_text(text)
    spans = [
        match.start()
        for match in re.finditer(r"\b(?:SP|SV|PS|DS|SR|SW|EX|OH|PC|OT|IO|APC|FSSB)\d{2,5}\b", canonical)
    ]
    if not spans:
        return []

    segments = []
    for index, start in enumerate(spans):
        end = spans[index + 1] if index + 1 < len(spans) else len(canonical)
        segment = canonical[start:end].strip()
        if segment:
            segments.append(segment)
    return segments


def block_sort_key(block: Dict[str, Any]) -> tuple:
    return (
        block.get("page_no") or 0,
        1 if block.get("block_type") == "page_context" else 0,
        block.get("block_id") or "",
    )


def relink_blocks(blocks: List[Dict[str, Any]]) -> None:
    blocks.sort(key=block_sort_key)
    for index, block in enumerate(blocks):
        block["prev_block_id"] = blocks[index - 1]["block_id"] if index > 0 else None
        block["next_block_id"] = blocks[index + 1]["block_id"] if index + 1 < len(blocks) else None
        block["context_prev"] = [
            blocks[i]["block_id"]
            for i in range(max(0, index - 2), index)
            if blocks[i].get("block_type") != "page_context"
        ]
        block["context_next"] = [
            blocks[i]["block_id"]
            for i in range(index + 1, min(len(blocks), index + 3))
            if blocks[i].get("block_type") != "page_context"
        ]
        page_no = block.get("page_no")
        block["page_context_id"] = f"{block['doc_id']}_page_{page_no:04d}_context" if isinstance(page_no, int) and page_no > 0 else None
        if block.get("block_type") == "page_context":
            block["page_context_id"] = block.get("block_id")


def build_page_context_blocks(doc_id: str, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    page_rows: Dict[int, Dict[str, Any]] = {}
    for block in blocks:
        if block.get("block_type") == "page_context":
            continue

        page_no = block.get("page_no")
        if not isinstance(page_no, int) or page_no <= 0:
            continue

        row = page_rows.setdefault(
            page_no,
            {
                "raw_text_parts": [],
                "layout_refs": set(),
                "image_paths": [],
                "hierarchy": {"chapter": "", "section": "", "subsection": ""},
                "section_path": "",
                "parse_levels": set(),
            },
        )
        raw_text = str(block.get("raw_text") or "").strip()
        if raw_text:
            row["raw_text_parts"].append(raw_text)
        row["layout_refs"].update(ref for ref in (block.get("layout_refs") or []) if ref)
        if not row["image_paths"] and block.get("image_paths"):
            row["image_paths"] = list(block.get("image_paths") or [])
        if not row["section_path"] and block.get("section_path"):
            row["section_path"] = str(block.get("section_path") or "")
            hierarchy = block.get("hierarchy") or {}
            row["hierarchy"] = {
                "chapter": str(hierarchy.get("chapter") or ""),
                "section": str(hierarchy.get("section") or ""),
                "subsection": str(hierarchy.get("subsection") or ""),
            }
        parse_level = str(block.get("parse_level") or "").strip()
        if parse_level:
            row["parse_levels"].add(parse_level)

    context_blocks: List[Dict[str, Any]] = []
    for page_no, row in sorted(page_rows.items()):
        raw_text = "\n".join(row["raw_text_parts"]).strip()
        if not raw_text:
            continue

        block = make_block(
            doc_id,
            "page_context",
            page_no,
            raw_text,
            row["hierarchy"],
            sorted(row["layout_refs"]),
            row["image_paths"],
            page_no,
        )
        block["block_id"] = f"{doc_id}_page_{page_no:04d}_context"
        block["block_level"] = 1
        block["title"] = f"page_{page_no:04d}_context"
        block["source"] = "page_context_aggregate"
        parse_levels = sorted(row["parse_levels"])
        block["parse_level"] = parse_levels[0] if len(parse_levels) == 1 else "merged"
        block["page_context_id"] = block["block_id"]
        block["searchable_text"] = make_searchable_text(block)
        context_blocks.append(block)
    return context_blocks


def build_blocks(doc_id: str, layout_json_path: Path) -> List[Dict[str, Any]]:
    data = read_json(layout_json_path)
    layout_items: List[Dict[str, Any]] = []
    walk_layout(data, layout_items)
    layout_items.sort(key=lambda item: (item.get("page_no") or 0, item.get("layout_ref") or ""))

    blocks: List[Dict[str, Any]] = []
    hierarchy = {"chapter": "", "section": "", "subsection": ""}
    block_index = 1

    for item in layout_items:
        page_no = item.get("page_no") if isinstance(item.get("page_no"), int) else -1
        raw_text = (item.get("text") or "").strip()
        layout_type = item.get("layout_type") or "text"
        layout_refs = [item.get("layout_ref") or ""]
        image_paths = [f"data/images/{doc_id}/page_{page_no:04d}.png"] if page_no > 0 else []

        if raw_text and is_heading(raw_text):
            hierarchy = update_hierarchy(hierarchy, raw_text)

        lower_type = str(layout_type).lower()
        if not raw_text and ("picture" in lower_type or "image" in lower_type or "figure" in lower_type):
            block = make_block(
                doc_id,
                "image",
                page_no,
                "",
                hierarchy,
                layout_refs,
                image_paths,
                block_index,
                item.get("bbox"),
                item.get("ocr_confidence"),
            )
            mark_refine_need(block)
            blocks.append(block)
            block_index += 1
            continue

        if not raw_text:
            continue

        block_type = "alarm_candidate" if extract_alarm_codes(raw_text) else classify_text(raw_text, layout_type)

        if block_type == "alarm_candidate":
            alarm_segments = split_alarm_segments(raw_text) or [raw_text]
            for segment in alarm_segments:
                block = make_block(
                    doc_id,
                    "alarm_candidate",
                    page_no,
                    segment,
                    hierarchy,
                    layout_refs,
                    image_paths,
                    block_index,
                    item.get("bbox"),
                    item.get("ocr_confidence"),
                )
                mark_refine_need(block)
                blocks.append(block)
                block_index += 1
            continue

        max_chars = PROCEDURE_CHUNK_RANGE[1] if block_type == "procedure" else TEXT_CHUNK_RANGE[1]
        if block_type not in {"procedure", "text"}:
            max_chars = MAX_BLOCK_CHARS
        for chunk in chunk_text(raw_text, min(max_chars, MAX_BLOCK_CHARS)):
            block = make_block(
                doc_id,
                block_type,
                page_no,
                chunk,
                hierarchy,
                layout_refs,
                image_paths,
                block_index,
                item.get("bbox"),
                item.get("ocr_confidence"),
            )
            mark_refine_need(block)
            blocks.append(block)
            block_index += 1

    apply_page_refine_flags(blocks)
    blocks.extend(build_page_context_blocks(doc_id, blocks))
    relink_blocks(blocks)
    return blocks


def main(doc_id: str) -> None:
    layout_json_path = DATA_DIR / "parsed" / doc_id / "layout.json"
    if not layout_json_path.exists():
        raise FileNotFoundError(f"Not found: {layout_json_path}")

    blocks = build_blocks(doc_id, layout_json_path)
    out_dir = ensure_dir(DATA_DIR / "blocks" / doc_id)
    draft_path = out_dir / "draft_blocks.jsonl"
    final_path = out_dir / "final_blocks.jsonl"
    compat_path = out_dir / "blocks.jsonl"
    write_jsonl(draft_path, blocks)
    write_jsonl(final_path, blocks)
    write_jsonl(compat_path, blocks)

    print(f"[OK] blocks: {len(blocks)}")
    print(f"[OK] draft: {draft_path}")
    print(f"[OK] final: {final_path}")
    print(f"[OK] compatibility: {compat_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/02_build_blocks.py <doc_id>")
        sys.exit(1)
    main(sys.argv[1])
