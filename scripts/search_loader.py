import importlib.util
from pathlib import Path


def load_search_engine(doc_id: str):
    script_path = Path(__file__).with_name("03_search_demo.py")
    spec = importlib.util.spec_from_file_location("search_demo", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import search script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_engine(doc_id)
