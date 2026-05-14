import argparse
import importlib.util
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from common import DATA_DIR, SOURCE_PRIORITIES, extract_alarm_codes, make_searchable_text, read_jsonl, utc_now_iso, write_jsonl


def load_build_module():
    script_path = Path(__file__).with_name("02_build_blocks.py")
    spec = importlib.util.spec_from_file_location("build_blocks_script", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import build script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def split_ocr_paragraphs(text: str) -> List[str]:
    text = str(text or "").strip()
    if not text:
        return []
    chunks = [part.strip() for part in re.split(r"\n{2,}", text) if part.strip()]
    if chunks:
        return chunks

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return []

    paragraphs: List[str] = []
    current = ""
    for line in lines:
        starts_new = bool(re.match(r"^(?:STEP\s*\d+|\d+[\.\)、]|[一二三四五六七八九十]+[、.])\s*", line, re.IGNORECASE))
        if starts_new and current:
            paragraphs.append(current.strip())
            current = line
            continue
        if not current:
            current = line
            continue
        if len(current) < 100:
            current = f"{current}\n{line}"
        else:
            paragraphs.append(current.strip())
            current = line
    if current:
        paragraphs.append(current.strip())
    return paragraphs


def normalize_block(block: Dict[str, Any], source_page_row: Dict[str, Any], page_context_id: str) -> Dict[str, Any]:
    block = dict(block)
    block_type = str(block.get("block_type") or "text")
    if block_type == "alarm_candidate":
        block_type = "alarm"
    elif block_type == "table_candidate":
        block_type = "table"

    timestamp = utc_now_iso()
    block["block_type"] = block_type
    block["block_level"] = 2 if block_type == "alarm" else block.get("block_level", 3)
    block["source_priority"] = SOURCE_PRIORITIES.get(block_type, block.get("source_priority", 0.5))
    block["alarm_codes"] = extract_alarm_codes(str(block.get("canonical_text") or block.get("raw_text") or ""))
    block["image_paths"] = [source_page_row["image_path"]]
    block["has_image"] = True
    block["bbox"] = None
    block["ocr_confidence"] = 1.0
    block["source"] = "dashscope_ocr"
    block["parse_level"] = "ocr_fixed"
    block["needs_refine"] = False
    block["refine_reason"] = []
    block["page_context_id"] = page_context_id
    block["layout_refs"] = list(block.get("layout_refs") or [])
    block["updated_at"] = timestamp
    block.setdefault("created_at", timestamp)
    block["searchable_text"] = make_searchable_text(block)
    return block


def build_ocr_page_blocks(
    build_module,
    doc_id: str,
    page_no: int,
    ocr_text: str,
    seed_block: Dict[str, Any] | None,
    local_index_start: int = 1,
) -> List[Dict[str, Any]]:
    hierarchy = dict((seed_block or {}).get("hierarchy") or {"chapter": "", "section": "", "subsection": ""})
    image_path = f"data/images/{doc_id}/page_{page_no:04d}.png"
    paragraphs = split_ocr_paragraphs(ocr_text)
    blocks: List[Dict[str, Any]] = []
    local_index = local_index_start
    page_context_id = f"{doc_id}_page_{page_no:04d}_context"

    for para_index, paragraph in enumerate(paragraphs, start=1):
        raw_text = paragraph.strip()
        if not raw_text:
            continue
        if build_module.is_heading(raw_text):
            hierarchy = build_module.update_hierarchy(hierarchy, raw_text)

        block_type = "alarm_candidate" if extract_alarm_codes(raw_text) else build_module.classify_text(raw_text, "text")
        candidate_texts = build_module.split_alarm_segments(raw_text) if block_type == "alarm_candidate" else [raw_text]
        for candidate_index, candidate_text in enumerate(candidate_texts, start=1):
            if not candidate_text.strip():
                continue
            block = build_module.make_block(
                doc_id,
                block_type,
                page_no,
                candidate_text.strip(),
                hierarchy,
                [f"ocr_page_{page_no:04d}:{para_index}:{candidate_index}"],
                [image_path],
                local_index,
                bbox=None,
                ocr_confidence=1.0,
            )
            block["block_id"] = f"{doc_id}_ocr{page_no:04d}_{local_index:04d}"
            block = normalize_block(block, {"image_path": image_path}, page_context_id)
            blocks.append(block)
            local_index += 1

    return blocks


def fix_blocks_with_ocr(final_blocks_path: str | Path, ocr_pages_path: str | Path) -> Tuple[Dict[str, Any], Path]:
    final_blocks_file = Path(final_blocks_path)
    ocr_pages_file = Path(ocr_pages_path)
    if not final_blocks_file.exists():
        raise FileNotFoundError(f"Not found: {final_blocks_file}")
    if not ocr_pages_file.exists():
        raise FileNotFoundError(f"Not found: {ocr_pages_file}")

    build_module = load_build_module()
    blocks = read_jsonl(final_blocks_file)
    ocr_pages = {
        int(row["page_no"]): row
        for row in read_jsonl(ocr_pages_file)
        if int(row.get("page_no") or 0) > 0 and str(row.get("ocr_text") or "").strip()
    }
    if not blocks:
        raise ValueError(f"No blocks found: {final_blocks_file}")
    if not ocr_pages:
        raise ValueError(f"No OCR pages found: {ocr_pages_file}")

    doc_id = str(blocks[0].get("doc_id") or final_blocks_file.parent.name)
    replaced_pages = sorted(ocr_pages)
    kept_blocks: List[Dict[str, Any]] = []
    page_seed: Dict[int, Dict[str, Any]] = {}

    for block in blocks:
        page_no = block.get("page_no")
        if isinstance(page_no, int) and page_no in ocr_pages and block.get("block_type") != "page_context":
            page_seed.setdefault(page_no, block)
            continue
        if block.get("block_type") == "page_context":
            continue
        kept_blocks.append(dict(block))

    new_blocks: List[Dict[str, Any]] = []
    for page_no in replaced_pages:
        new_blocks.extend(
            build_ocr_page_blocks(
                build_module,
                doc_id,
                page_no,
                str(ocr_pages[page_no]["ocr_text"]),
                page_seed.get(page_no),
            )
        )

    final_non_context = kept_blocks + new_blocks
    final_non_context.sort(key=build_module.block_sort_key)
    final_blocks = final_non_context + build_module.build_page_context_blocks(doc_id, final_non_context)
    build_module.relink_blocks(final_blocks)

    block_dir = DATA_DIR / "blocks" / doc_id
    out_path = block_dir / "final_blocks_ocr_fixed.jsonl"
    write_jsonl(out_path, final_blocks)

    summary = {
        "doc_id": doc_id,
        "source_final_blocks": str(final_blocks_file.resolve()),
        "source_ocr_pages": str(ocr_pages_file.resolve()),
        "replaced_pages": replaced_pages,
        "replaced_page_count": len(replaced_pages),
        "final_block_count": len(final_blocks),
        "output_path": str(out_path),
    }
    return summary, out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Replace garbled pages in final_blocks.jsonl with DashScope OCR text blocks.")
    parser.add_argument("final_blocks_jsonl")
    parser.add_argument("ocr_pages_jsonl")
    args = parser.parse_args()

    summary, out_path = fix_blocks_with_ocr(args.final_blocks_jsonl, args.ocr_pages_jsonl)
    print(f"[OK] replaced pages: {summary['replaced_page_count']}")
    print(f"[OK] final blocks: {summary['final_block_count']}")
    print(f"[OK] output: {out_path}")


if __name__ == "__main__":
    main()
