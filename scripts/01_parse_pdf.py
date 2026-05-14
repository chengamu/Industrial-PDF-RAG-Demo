import sys
from pathlib import Path

import fitz
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter
from docling.document_converter import PdfFormatOption

from common import DATA_DIR, ensure_dir, file_sha256, slugify_doc_id, write_json, write_jsonl


def render_page_images(pdf_path: Path, image_dir: Path, zoom: float = 2.0) -> int:
    doc = fitz.open(pdf_path)
    matrix = fitz.Matrix(zoom, zoom)
    try:
        for page_index, page in enumerate(doc, start=1):
            pix = page.get_pixmap(matrix=matrix)
            img_path = image_dir / f"page_{page_index:04d}.png"
            pix.save(str(img_path))
        return len(doc)
    finally:
        doc.close()


def extract_pages_text(pdf_path: Path, output_path: Path) -> int:
    doc = fitz.open(pdf_path)
    rows = []
    try:
        for page_index, page in enumerate(doc, start=1):
            rows.append(
                {
                    "page_no": page_index,
                    "text": page.get_text("text") or "",
                }
            )
    finally:
        doc.close()
    write_jsonl(output_path, rows)
    return len(rows)


def parse_pdf(pdf_path: str, doc_id: str | None = None) -> str:
    pdf_path = Path(pdf_path).resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc_id = doc_id or slugify_doc_id(pdf_path.stem)
    parsed_dir = ensure_dir(DATA_DIR / "parsed" / doc_id)
    image_dir = ensure_dir(DATA_DIR / "images" / doc_id)

    print(f"[1] Render page images: {pdf_path}")
    page_count = render_page_images(pdf_path, image_dir)

    pages_text_path = parsed_dir / "pages_text.jsonl"
    print("[2] Extract page text fallback with PyMuPDF...")
    extract_pages_text(pdf_path, pages_text_path)

    print("[3] Parse PDF with Docling...")
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = False
    pipeline_options.force_backend_text = True
    pipeline_options.layout_batch_size = 1
    pipeline_options.table_batch_size = 1
    pipeline_options.ocr_batch_size = 1
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )
    result = converter.convert(str(pdf_path))
    document = result.document

    md_path = parsed_dir / "document.md"
    md_path.write_text(document.export_to_markdown(), encoding="utf-8")

    layout_path = parsed_dir / "layout.json"
    write_json(layout_path, document.export_to_dict())

    metadata_path = parsed_dir / "metadata.json"
    write_json(
        metadata_path,
        {
            "doc_id": doc_id,
            "source_pdf": str(pdf_path),
            "original_filename": pdf_path.name,
            "page_count": page_count,
            "sha256": file_sha256(pdf_path),
        },
    )

    print(f"[OK] doc_id: {doc_id}")
    print(f"[OK] Markdown: {md_path}")
    print(f"[OK] Layout JSON: {layout_path}")
    print(f"[OK] Page text: {pages_text_path}")
    print(f"[OK] Images: {image_dir}")
    return doc_id


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/01_parse_pdf.py <pdf_path> [doc_id]")
        sys.exit(1)

    parse_pdf(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
