import sys

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def main() -> None:
    script_path = Path(__file__).with_name("01_parse_pdf.py")
    spec = spec_from_file_location("parse_pdf_script", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import parser script: {script_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    if len(sys.argv) < 2:
        print("Usage: python scripts/01_parse_pdf_light.py <pdf_path> [doc_id]")
        sys.exit(1)
    module.parse_pdf(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)


if __name__ == "__main__":
    main()
