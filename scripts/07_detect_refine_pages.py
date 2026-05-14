import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from common import DATA_DIR, ensure_dir, read_json, read_jsonl, write_json


def parse_pages(value: str | None) -> List[int]:
    if not value:
        return []
    pages: List[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = [int(x.strip()) for x in part.split("-", 1)]
            pages.extend(range(start, end + 1))
        else:
            pages.append(int(part))
    return sorted(set(page for page in pages if page > 0))


def load_page_count(doc_id: str) -> int:
    metadata_path = DATA_DIR / "parsed" / doc_id / "metadata.json"
    if not metadata_path.exists():
        return 0
    metadata = read_json(metadata_path)
    return int(metadata.get("page_count") or 0)


def detect_refine_pages(
    doc_id: str,
    include_low_text: bool = False,
    manual_pages: List[int] | None = None,
    low_text_chars: int = 80,
    top: int | None = None,
) -> Dict[str, Any]:
    block_path = DATA_DIR / "blocks" / doc_id / "draft_blocks.jsonl"
    if not block_path.exists():
        block_path = DATA_DIR / "blocks" / doc_id / "blocks.jsonl"
    if not block_path.exists():
        raise FileNotFoundError(f"Not found: {block_path}")

    blocks = read_jsonl(block_path)
    page_count = load_page_count(doc_id)
    page_stats: Dict[int, Dict[str, Any]] = defaultdict(
        lambda: {"page_no": 0, "reasons": set(), "block_count": 0, "text_chars": 0, "alarm_count": 0}
    )

    for block in blocks:
        page_no = block.get("page_no")
        if not isinstance(page_no, int) or page_no <= 0:
            continue
        stat = page_stats[page_no]
        stat["page_no"] = page_no
        stat["block_count"] += 1
        stat["text_chars"] += len(str(block.get("raw_text") or ""))
        stat["alarm_count"] += len(block.get("alarm_codes") or [])
        for reason in block.get("refine_reason") or []:
            stat["reasons"].add(reason)
        if block.get("needs_refine"):
            stat["reasons"].add("block_needs_refine")

    for page_no in manual_pages or []:
        stat = page_stats[page_no]
        stat["page_no"] = page_no
        stat["reasons"].add("manual")

    if include_low_text and page_count:
        for page_no in range(1, page_count + 1):
            stat = page_stats[page_no]
            stat["page_no"] = page_no
            if stat["text_chars"] < low_text_chars:
                stat["reasons"].add("low_text")

    rows = []
    for page_no, stat in page_stats.items():
        reasons = sorted(stat["reasons"])
        if not reasons:
            continue
        rows.append(
            {
                "page_no": page_no,
                "reasons": reasons,
                "block_count": stat["block_count"],
                "text_chars": stat["text_chars"],
                "alarm_count": stat["alarm_count"],
            }
        )

    reason_priority = {
        "manual": 100,
        "alarm_dense": 90,
        "alarm_candidate": 80,
        "table_candidate": 70,
        "low_text": 50,
        "image": 40,
        "block_needs_refine": 30,
    }

    def row_score(row: Dict[str, Any]) -> tuple:
        score = max(reason_priority.get(reason, 0) for reason in row["reasons"])
        return (-score, row["page_no"])

    rows.sort(key=row_score)
    if top is not None:
        rows = rows[:top]

    report = {
        "doc_id": doc_id,
        "page_count": page_count,
        "refine_page_count": len(rows),
        "pages": rows,
    }
    out_path = ensure_dir(DATA_DIR / "refine" / doc_id) / "refine_pages.json"
    write_json(out_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect pages that need second-pass refinement.")
    parser.add_argument("doc_id")
    parser.add_argument("--pages", default="", help="Manual pages, e.g. 622 or 600-630,640")
    parser.add_argument("--include-low-text", action="store_true")
    parser.add_argument("--low-text-chars", type=int, default=80)
    parser.add_argument("--top", type=int, default=None)
    args = parser.parse_args()

    report = detect_refine_pages(
        args.doc_id,
        include_low_text=args.include_low_text,
        manual_pages=parse_pages(args.pages),
        low_text_chars=args.low_text_chars,
        top=args.top,
    )
    print(f"[OK] refine pages: {report['refine_page_count']}")
    print(f"[OK] output: {DATA_DIR / 'refine' / args.doc_id / 'refine_pages.json'}")
    for row in report["pages"][:20]:
        print(f"page={row['page_no']} reasons={','.join(row['reasons'])} chars={row['text_chars']} alarms={row['alarm_count']}")


if __name__ == "__main__":
    main()
