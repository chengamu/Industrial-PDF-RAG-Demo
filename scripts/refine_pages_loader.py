import importlib.util
from pathlib import Path


def parse_pages(value: str | None):
    script_path = Path(__file__).with_name("07_detect_refine_pages.py")
    spec = importlib.util.spec_from_file_location("detect_refine_pages", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import refine detector: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.parse_pages(value)
