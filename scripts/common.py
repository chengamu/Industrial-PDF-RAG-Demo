import hashlib
import importlib.util
import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

ALARM_PREFIXES = ("SP", "SV", "PS", "DS", "SR", "SW", "EX", "OH", "PC", "OT", "IO", "APC", "FSSB")
ALARM_CODE_PATTERN = re.compile(
    r"\b(?:SP|SV|PS|DS|SR|SW|EX|OH|PC|OT|IO|APC|FSSB)\s*[-_]?\s*\d{2,5}\b",
    re.IGNORECASE,
)
SOURCE_PRIORITIES = {
    "alarm": 1.0,
    "alarm_candidate": 0.95,
    "warning": 0.9,
    "procedure": 0.8,
    "table": 0.7,
    "table_candidate": 0.7,
    "image": 0.6,
    "page_context": 0.3,
    "text": 0.5,
}

MAX_BLOCK_CHARS = 1200
PROCEDURE_CHUNK_RANGE = (300, 800)
TEXT_CHUNK_RANGE = (500, 1000)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def slugify_doc_id(name: str) -> str:
    normalized = unicodedata.normalize("NFKC", name)
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", normalized).strip("_")
    return normalized or "document"


def normalize_alarm_code(code: str) -> str:
    return re.sub(r"[\s\-_]", "", code.upper())


def _spaced_prefix_pattern(prefix: str) -> str:
    return r"\s*".join(re.escape(ch) for ch in prefix)


def iter_alarm_spans(text: str) -> List[Tuple[int, int, str]]:
    spans: List[Tuple[int, int, str]] = []
    for prefix in ALARM_PREFIXES:
        pattern = re.compile(
            rf"\b{_spaced_prefix_pattern(prefix)}\s*[-_]?\s*(\d(?:\s*\d){{1,4}})\b",
            re.IGNORECASE,
        )
        for match in pattern.finditer(text or ""):
            digits = re.sub(r"\D", "", match.group(1))
            if 2 <= len(digits) <= 5:
                spans.append((match.start(), match.end(), f"{prefix}{digits}"))
    spans.sort(key=lambda item: item[0])

    deduped: List[Tuple[int, int, str]] = []
    last_end = -1
    for span in spans:
        if span[0] < last_end:
            continue
        deduped.append(span)
        last_end = span[1]
    return deduped


def extract_alarm_codes(text: str) -> List[str]:
    codes = [code for _, _, code in iter_alarm_spans(text or "")]
    return sorted(set(normalize_alarm_code(code) for code in codes))


def canonicalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")

    def replace_alarm(match: re.Match[str]) -> str:
        raw = match.group(0)
        codes = extract_alarm_codes(raw)
        return codes[0] if codes else raw

    for prefix in ALARM_PREFIXES:
        pattern = re.compile(
            rf"\b{_spaced_prefix_pattern(prefix)}\s*[-_]?\s*\d(?:\s*\d){{1,4}}\b",
            re.IGNORECASE,
        )
        text = pattern.sub(replace_alarm, text)

    text = text.upper()
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def make_searchable_text(block: Dict[str, Any]) -> str:
    parts = [
        block.get("title", ""),
        " ".join(block.get("alarm_codes") or []),
        block.get("section_path", ""),
        block.get("canonical_text", ""),
    ]
    return "\n".join(str(part).strip() for part in parts if str(part).strip())


def tokenize(text: str) -> List[str]:
    text = canonicalize_text(text)
    tokens = re.findall(r"[A-Z]+[0-9]+|[A-Z]+|[0-9]+|[\u4e00-\u9fff]", text)
    return tokens


def parse_query(query: str) -> Dict[str, Any]:
    canonical_query = canonicalize_text(query)
    alarm_codes = extract_alarm_codes(canonical_query)
    alarm_set = set(alarm_codes)
    keywords = [token for token in tokenize(canonical_query) if token not in alarm_set]
    return {
        "alarm_codes": alarm_codes,
        "keywords": keywords,
        "canonical_query": canonical_query,
    }


def normalize_optional_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_dotenv_if_available() -> None:
    if importlib.util.find_spec("dotenv") is None:
        return
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
