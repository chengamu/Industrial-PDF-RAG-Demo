import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from common import DATA_DIR, read_jsonl, write_json


SUSPECT_TOKENS = [
    "鍙",
    "缁",
    "涔",
    "鐨",
    "鎴",
    "鎿",
    "灏",
    "鍦",
    "閫",
    "浣",
    "鐩",
    "瀛",
    "銆",
    "锛",
    "锟",
    "€",
    "�",
    "鈥",
]


def page_text_stats(text: str) -> Dict[str, Any]:
    text = str(text or "")
    total_chars = len(text)
    question_count = text.count("?")
    suspect_hits = sum(text.count(token) for token in SUSPECT_TOKENS)
    ascii_letters = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    digits = sum(1 for ch in text if ch.isdigit())
    cjk_chars = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    cyrillic_chars = sum(1 for ch in text if "\u0400" <= ch <= "\u04ff")
    empty_like = 1 if not text.strip() else 0
    return {
        "total_chars": total_chars,
        "question_count": question_count,
        "suspect_hits": suspect_hits,
        "ascii_letters": ascii_letters,
        "digits": digits,
        "cjk_chars": cjk_chars,
        "cyrillic_chars": cyrillic_chars,
        "empty_like": empty_like,
    }


def score_page(page_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    joined = "\n".join(str(row.get("raw_text") or "") for row in page_rows if row.get("block_type") != "page_context")
    stats = page_text_stats(joined)
    score = 0.0
    reasons: List[str] = []

    if stats["suspect_hits"] >= 2:
        score += min(4.0, stats["suspect_hits"] * 0.8)
        reasons.append("suspect_tokens")
    if stats["cyrillic_chars"] >= 2:
        score += min(4.0, stats["cyrillic_chars"] * 0.6)
        reasons.append("cyrillic_chars")
    if stats["question_count"] >= 2:
        score += min(3.0, stats["question_count"] * 0.5)
        reasons.append("question_marks")
    if 0 < stats["total_chars"] <= 24:
        score += 1.0
        reasons.append("very_short_page_text")
    if stats["empty_like"]:
        score += 1.0
        reasons.append("empty_text")

    return {
        "score": round(score, 3),
        "reasons": reasons,
        "stats": stats,
        "text_preview": joined[:240],
    }


def detect_garbled_pages(final_blocks_path: str | Path, min_score: float = 2.0, top: int | None = None) -> Tuple[Dict[str, Any], Path]:
    final_blocks_file = Path(final_blocks_path)
    if not final_blocks_file.exists():
        raise FileNotFoundError(f"Not found: {final_blocks_file}")

    blocks = read_jsonl(final_blocks_file)
    if not blocks:
        raise ValueError(f"No blocks found: {final_blocks_file}")

    doc_id = str(blocks[0].get("doc_id") or final_blocks_file.parent.name)
    by_page: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for block in blocks:
        page_no = block.get("page_no")
        if isinstance(page_no, int) and page_no > 0:
            by_page[page_no].append(block)

    rows: List[Dict[str, Any]] = []
    for page_no, page_rows in sorted(by_page.items()):
        scored = score_page(page_rows)
        if scored["score"] < min_score:
            continue
        image_paths = []
        for row in page_rows:
            for path in row.get("image_paths") or []:
                if path and path not in image_paths:
                    image_paths.append(path)
        rows.append(
            {
                "page_no": page_no,
                "score": scored["score"],
                "reasons": scored["reasons"],
                "stats": scored["stats"],
                "image_paths": image_paths,
                "text_preview": scored["text_preview"],
            }
        )

    rows.sort(key=lambda row: (-row["score"], row["page_no"]))
    if top is not None:
        rows = rows[:top]

    report = {
        "doc_id": doc_id,
        "final_blocks_jsonl": str(final_blocks_file.resolve()),
        "garbled_page_count": len(rows),
        "pages": rows,
    }
    out_path = DATA_DIR / "ocr" / doc_id / "garbled_pages.json"
    write_json(out_path, report)
    return report, out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect likely garbled pages from final_blocks.jsonl.")
    parser.add_argument("final_blocks_jsonl")
    parser.add_argument("--min-score", type=float, default=2.0)
    parser.add_argument("--top", type=int, default=None)
    args = parser.parse_args()

    report, out_path = detect_garbled_pages(args.final_blocks_jsonl, args.min_score, args.top)
    print(f"[OK] garbled pages: {report['garbled_page_count']}")
    print(f"[OK] output: {out_path}")
    for row in report["pages"][:20]:
        print(f"page={row['page_no']} score={row['score']:.2f} reasons={','.join(row['reasons'])} preview={row['text_preview'][:80]}")


if __name__ == "__main__":
    main()
