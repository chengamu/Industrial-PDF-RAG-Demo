import argparse
import base64
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from openai import OpenAI

from common import DATA_DIR, load_dotenv_if_available, read_json, read_jsonl, utc_now_iso, write_jsonl


DEFAULT_OCR_MODEL = os.getenv("DASHSCOPE_OCR_MODEL", "qwen-vl-ocr-latest")
DEFAULT_BASE_URL = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
DEFAULT_PROMPT = "提取图片中的全部文字，按原阅读顺序输出纯文本。不要解释，不要总结，不要补充。"


def parse_pages(value: str | None) -> List[int]:
    if not value:
        return []
    pages: List[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = [int(item.strip()) for item in part.split("-", 1)]
            pages.extend(range(start, end + 1))
        else:
            pages.append(int(part))
    return sorted(set(page for page in pages if page > 0))


def resolve_pages(doc_id: str, report_path: str | None, manual_pages: List[int]) -> List[Dict[str, Any]]:
    pages = {page_no: {"page_no": page_no} for page_no in manual_pages}
    if report_path:
        report_file = Path(report_path)
    else:
        report_file = DATA_DIR / "ocr" / doc_id / "garbled_pages.json"

    if report_file.exists():
        report = read_json(report_file)
        for row in report.get("pages", []):
            page_no = int(row.get("page_no") or 0)
            if page_no <= 0:
                continue
            pages[page_no] = {
                "page_no": page_no,
                "score": row.get("score"),
                "reasons": row.get("reasons") or [],
            }

    return [pages[page_no] for page_no in sorted(pages)]


def image_to_data_url(image_path: Path) -> str:
    suffix = image_path.suffix.lower().lstrip(".") or "png"
    mime = "jpeg" if suffix in {"jpg", "jpeg"} else suffix
    b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:image/{mime};base64,{b64}"


def load_existing_rows(out_path: Path) -> Dict[int, Dict[str, Any]]:
    if not out_path.exists():
        return {}
    return {
        int(row["page_no"]): row
        for row in read_jsonl(out_path)
        if int(row.get("page_no") or 0) > 0
    }


def create_client() -> Tuple[OpenAI, str]:
    load_dotenv_if_available()
    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing DASHSCOPE_API_KEY.")
    client = OpenAI(api_key=api_key, base_url=os.getenv("DASHSCOPE_BASE_URL", DEFAULT_BASE_URL))
    return client, api_key


def ocr_pages(
    doc_id: str,
    pages: List[Dict[str, Any]],
    model: str = DEFAULT_OCR_MODEL,
    force: bool = False,
) -> Tuple[List[Dict[str, Any]], Path]:
    image_dir = DATA_DIR / "images" / doc_id
    if not image_dir.exists():
        raise FileNotFoundError(f"Not found: {image_dir}")

    out_path = DATA_DIR / "ocr" / doc_id / "ocr_pages.jsonl"
    existing = load_existing_rows(out_path)
    client, _ = create_client()

    rows_by_page = dict(existing)
    for row in pages:
        page_no = int(row["page_no"])
        if page_no in rows_by_page and not force:
            continue

        image_path = image_dir / f"page_{page_no:04d}.png"
        if not image_path.exists():
            raise FileNotFoundError(f"Not found: {image_path}")

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": DEFAULT_PROMPT},
                        {"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}},
                    ],
                }
            ],
            temperature=0.01,
        )
        text = ""
        if response.choices:
            text = response.choices[0].message.content or ""

        rows_by_page[page_no] = {
            "doc_id": doc_id,
            "page_no": page_no,
            "image_path": f"data/images/{doc_id}/page_{page_no:04d}.png",
            "ocr_text": text.strip(),
            "model": model,
            "base_url": os.getenv("DASHSCOPE_BASE_URL", DEFAULT_BASE_URL),
            "garbled_score": row.get("score"),
            "garbled_reasons": row.get("reasons") or [],
            "created_at": utc_now_iso(),
        }
        print(f"[OCR] page={page_no} chars={len(text.strip())}")

    output_rows = [rows_by_page[page_no] for page_no in sorted(rows_by_page)]
    write_jsonl(out_path, output_rows)
    return output_rows, out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DashScope OCR on selected PDF page images.")
    parser.add_argument("doc_id")
    parser.add_argument("garbled_pages_json", nargs="?")
    parser.add_argument("--pages", default="")
    parser.add_argument("--model", default=DEFAULT_OCR_MODEL)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    pages = resolve_pages(args.doc_id, args.garbled_pages_json, parse_pages(args.pages))
    if not pages:
        raise ValueError("No pages to OCR. Pass --pages or provide garbled_pages.json.")

    rows, out_path = ocr_pages(args.doc_id, pages, model=args.model, force=args.force)
    print(f"[OK] ocr pages: {len(rows)}")
    print(f"[OK] output: {out_path}")


if __name__ == "__main__":
    main()
