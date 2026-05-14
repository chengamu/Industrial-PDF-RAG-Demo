import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List

import fitz
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from common import DATA_DIR, SOURCE_PRIORITIES, ensure_dir, make_searchable_text, read_json, read_jsonl, write_json, write_jsonl
from refine_pages_loader import parse_pages


def load_build_blocks():
    script_path = Path(__file__).with_name("02_build_blocks.py")
    spec = importlib.util.spec_from_file_location("build_blocks_script", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import build script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_blocks


def find_source_pdf(doc_id: str) -> Path:
    metadata_path = DATA_DIR / "parsed" / doc_id / "metadata.json"
    if metadata_path.exists():
        metadata = read_json(metadata_path)
        source_pdf = Path(str(metadata.get("source_pdf") or ""))
        if source_pdf.exists():
            return source_pdf

    pdfs = sorted((DATA_DIR / "raw").glob("*.pdf"))
    if len(pdfs) == 1:
        return pdfs[0]
    raise FileNotFoundError("Cannot resolve source PDF. Pass --pdf explicitly.")


def load_page_numbers(path: Path) -> List[int]:
    if not path.exists():
        return []
    data = read_json(path)
    return sorted(
        {
            int(row["page_no"])
            for row in data.get("pages", [])
            if int(row.get("page_no") or 0) > 0
        }
    )


def load_refine_pages(doc_id: str, refine_pages_path: str | None, manual_pages: List[int]) -> List[int]:
    pages = set(manual_pages)
    if refine_pages_path:
        candidate_paths = [Path(refine_pages_path)]
    else:
        refine_dir = DATA_DIR / "refine" / doc_id
        candidate_paths = [
            refine_dir / "refine_queue.json",
            refine_dir / "refine_pages.json",
        ]

    for path in candidate_paths:
        pages.update(load_page_numbers(path))

    if not pages:
        raise ValueError("No refine pages found. Build refine_queue.json / refine_pages.json first, or pass --pages.")
    return sorted(pages)


def make_single_page_pdf(source_pdf: Path, page_no: int, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    source = fitz.open(source_pdf)
    target = fitz.open()
    try:
        if page_no < 1 or page_no > len(source):
            raise ValueError(f"Page out of range: {page_no}")
        target.insert_pdf(source, from_page=page_no - 1, to_page=page_no - 1)
        target.save(out_path)
    finally:
        target.close()
        source.close()


def create_refine_converter() -> DocumentConverter:
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.force_backend_text = False
    pipeline_options.generate_table_images = False
    pipeline_options.generate_page_images = False
    pipeline_options.layout_batch_size = 1
    pipeline_options.table_batch_size = 1
    pipeline_options.ocr_batch_size = 1
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )


def normalize_refined_block(block: Dict[str, Any], doc_id: str, page_no: int, index: int) -> Dict[str, Any]:
    block = dict(block)
    refined_type = block.get("block_type")
    if refined_type == "alarm_candidate":
        refined_type = "alarm"
    elif refined_type == "table_candidate":
        refined_type = "table"

    block["block_id"] = f"{doc_id}_r{page_no:04d}_{index:04d}"
    block["doc_id"] = doc_id
    block["block_type"] = refined_type
    block["block_level"] = 2 if refined_type == "alarm" else block.get("block_level", 3)
    block["source_priority"] = SOURCE_PRIORITIES.get(refined_type, block.get("source_priority", 0.7))
    block["page_no"] = page_no
    block["image_paths"] = [f"data/images/{doc_id}/page_{page_no:04d}.png"]
    block["has_image"] = True
    block["layout_refs"] = [f"refined_page_{page_no:04d}:{ref}" for ref in block.get("layout_refs", [])]
    block["source"] = "refined_docling_page"
    block["needs_refine"] = False
    block["refine_reason"] = []
    block["parse_level"] = "refined"
    block["refined_from_page"] = page_no
    block["searchable_text"] = make_searchable_text(block)
    return block


def parse_refine_pages(doc_id: str, pages: List[int], source_pdf: Path, limit: int | None = None) -> List[Dict[str, Any]]:
    build_blocks = load_build_blocks()
    converter = create_refine_converter()
    base_dir = ensure_dir(DATA_DIR / "refine" / doc_id)
    out_path = base_dir / "refined_blocks.jsonl"
    selected_pages = pages[: limit or None]
    refined_blocks: List[Dict[str, Any]] = []
    if out_path.exists():
        refined_blocks = [
            block for block in read_jsonl(out_path)
            if block.get("page_no") not in set(selected_pages)
        ]

    new_blocks: List[Dict[str, Any]] = []
    for page_no in selected_pages:
        print(f"[REFINE] page {page_no}")
        page_pdf = base_dir / "pages" / f"page_{page_no:04d}.pdf"
        page_dir = ensure_dir(base_dir / "parsed" / f"page_{page_no:04d}")
        make_single_page_pdf(source_pdf, page_no, page_pdf)

        result = converter.convert(str(page_pdf))
        document = result.document
        (page_dir / "document.md").write_text(document.export_to_markdown(), encoding="utf-8")
        layout_path = page_dir / "layout.json"
        write_json(layout_path, document.export_to_dict())

        page_blocks = build_blocks(f"{doc_id}_page_{page_no:04d}", layout_path)
        for local_index, block in enumerate(page_blocks, start=1):
            if not str(block.get("raw_text") or "").strip() and block.get("block_type") != "image":
                continue
            new_blocks.append(normalize_refined_block(block, doc_id, page_no, local_index))

    refined_blocks.extend(new_blocks)
    write_jsonl(out_path, refined_blocks)
    print(f"[OK] new refined blocks: {len(new_blocks)}")
    print(f"[OK] total refined blocks: {len(refined_blocks)}")
    print(f"[OK] output: {out_path}")
    return refined_blocks


def main() -> None:
    parser = argparse.ArgumentParser(description="Run second-pass Docling parsing for selected pages.")
    parser.add_argument("doc_id")
    parser.add_argument("refine_pages_json", nargs="?")
    parser.add_argument("--pages", default="", help="Manual pages, e.g. 622 or 600-630,640")
    parser.add_argument("--pdf", default="")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    source_pdf = Path(args.pdf) if args.pdf else find_source_pdf(args.doc_id)
    pages = load_refine_pages(args.doc_id, args.refine_pages_json, parse_pages(args.pages))
    parse_refine_pages(args.doc_id, pages, source_pdf, args.limit)


if __name__ == "__main__":
    main()
