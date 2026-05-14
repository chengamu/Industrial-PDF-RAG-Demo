import argparse
from collections import defaultdict
from typing import Any, Dict, List

from common import DATA_DIR, SOURCE_PRIORITIES, canonicalize_text, extract_alarm_codes, make_searchable_text, read_jsonl, utc_now_iso, write_jsonl


REPLACED_DRAFT_TYPES = {"alarm_candidate", "table_candidate"}


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
        block.setdefault("created_at", utc_now_iso())
        block["updated_at"] = utc_now_iso()


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

        canonical_text = canonicalize_text(raw_text)
        timestamp = utc_now_iso()
        block = {
            "block_id": f"{doc_id}_page_{page_no:04d}_context",
            "doc_id": doc_id,
            "block_type": "page_context",
            "block_level": 1,
            "source_priority": SOURCE_PRIORITIES.get("page_context", 0.3),
            "page_no": page_no,
            "title": f"page_{page_no:04d}_context",
            "raw_text": raw_text,
            "canonical_text": canonical_text,
            "searchable_text": "",
            "alarm_codes": extract_alarm_codes(canonical_text),
            "section_path": row["section_path"],
            "hierarchy": row["hierarchy"],
            "prev_block_id": None,
            "next_block_id": None,
            "parent_block_id": None,
            "context_prev": [],
            "context_next": [],
            "page_context_id": f"{doc_id}_page_{page_no:04d}_context",
            "layout_refs": sorted(row["layout_refs"]),
            "bbox": None,
            "image_paths": row["image_paths"],
            "has_image": bool(row["image_paths"]),
            "ocr_confidence": None,
            "source": "page_context_aggregate",
            "needs_refine": False,
            "refine_reason": [],
            "parse_level": "",
            "created_at": timestamp,
            "updated_at": timestamp,
        }
        parse_levels = sorted(row["parse_levels"])
        block["parse_level"] = parse_levels[0] if len(parse_levels) == 1 else "merged"
        block["searchable_text"] = make_searchable_text(block)
        context_blocks.append(block)
    return context_blocks


def merge_blocks(doc_id: str) -> Dict[str, Any]:
    block_dir = DATA_DIR / "blocks" / doc_id
    refine_dir = DATA_DIR / "refine" / doc_id
    draft_path = block_dir / "draft_blocks.jsonl"
    refined_path = refine_dir / "refined_blocks.jsonl"
    if not draft_path.exists():
        raise FileNotFoundError(f"Not found: {draft_path}")
    if not refined_path.exists():
        raise FileNotFoundError(f"Not found: {refined_path}")

    draft_blocks = read_jsonl(draft_path)
    refined_blocks = read_jsonl(refined_path)
    refined_by_page = defaultdict(list)
    for block in refined_blocks:
        page_no = block.get("page_no")
        if isinstance(page_no, int):
            refined_by_page[page_no].append(block)

    final_blocks: List[Dict[str, Any]] = []
    replaced_count = 0
    for block in draft_blocks:
        if block.get("block_type") == "page_context":
            continue
        page_no = block.get("page_no")
        if page_no in refined_by_page and block.get("block_type") in REPLACED_DRAFT_TYPES:
            replaced_count += 1
            continue
        final_blocks.append(block)

    for page_blocks in refined_by_page.values():
        final_blocks.extend(block for block in page_blocks if block.get("block_type") != "page_context")

    final_blocks.extend(build_page_context_blocks(doc_id, final_blocks))
    relink_blocks(final_blocks)
    final_path = block_dir / "final_blocks.jsonl"
    compat_path = block_dir / "blocks.jsonl"
    write_jsonl(final_path, final_blocks)
    write_jsonl(compat_path, final_blocks)

    return {
        "doc_id": doc_id,
        "draft_count": len(draft_blocks),
        "refined_count": len(refined_blocks),
        "replaced_draft_count": replaced_count,
        "final_count": len(final_blocks),
        "refined_pages": sorted(refined_by_page.keys()),
        "final_path": str(final_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge draft and refined blocks into final_blocks.jsonl.")
    parser.add_argument("doc_id")
    args = parser.parse_args()

    summary = merge_blocks(args.doc_id)
    print(f"[OK] draft blocks: {summary['draft_count']}")
    print(f"[OK] refined blocks: {summary['refined_count']}")
    print(f"[OK] replaced draft blocks: {summary['replaced_draft_count']}")
    print(f"[OK] final blocks: {summary['final_count']}")
    print(f"[OK] refined pages: {summary['refined_pages']}")
    print(f"[OK] final: {summary['final_path']}")


if __name__ == "__main__":
    main()
