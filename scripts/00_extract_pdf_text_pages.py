import sys
from pathlib import Path

from common import DATA_DIR, ensure_dir, slugify_doc_id
from importlib.util import module_from_spec, spec_from_file_location


def load_extract_pages_text():
    script_path = Path(__file__).with_name("01_parse_pdf.py")
    spec = spec_from_file_location("parse_pdf_script", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import parser script: {script_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.extract_pages_text


def main(pdf_path: str, doc_id: str | None = None) -> None:
    pdf = Path(pdf_path).resolve()
    if not pdf.exists():
        raise FileNotFoundError(f"PDF not found: {pdf}")
    doc_id = doc_id or slugify_doc_id(pdf.stem)
    parsed_dir = ensure_dir(DATA_DIR / "parsed" / doc_id)
    out_path = parsed_dir / "pages_text.jsonl"
    count = load_extract_pages_text()(pdf, out_path)
    print(f"[OK] pages: {count}")
    print(f"[OK] output: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/00_extract_pdf_text_pages.py <pdf_path> [doc_id]")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
