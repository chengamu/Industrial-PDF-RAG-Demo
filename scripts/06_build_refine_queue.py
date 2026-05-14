import argparse
from pathlib import Path
from typing import Any, Dict, List

from common import DATA_DIR, ensure_dir, read_json, write_json


def add_page(queue: Dict[int, Dict[str, Any]], page_no: int, reason: str, query: str) -> None:
    row = queue.setdefault(page_no, {"page_no": page_no, "reason": [], "queries": []})
    if reason not in row["reason"]:
        row["reason"].append(reason)
    if query and query not in row["queries"]:
        row["queries"].append(query)


def add_missing_coverage_pages(
    queue: Dict[int, Dict[str, Any]],
    missing_coverage_path: Path,
    min_pages_text_score: float,
) -> int:
    if not missing_coverage_path.exists():
        return 0

    report = read_json(missing_coverage_path)
    added = 0
    for item in report.get("items", []):
        score = float(item.get("page_score") or 0.0)
        if score < min_pages_text_score:
            continue

        query = str(item.get("query") or "")
        for page_no in item.get("suggest_refine_pages") or [item.get("page_no")]:
            if not page_no:
                continue
            before = len(queue)
            add_page(queue, int(page_no), "pages_text_coverage_missing", query)
            if len(queue) > before:
                added += 1
    return added


def build_queue(
    doc_id: str,
    failure_report_path: str | None = None,
    missing_coverage_path: str | None = None,
    min_pages_text_score: float = 15.0,
) -> Dict[str, Any]:
    report_path = Path(failure_report_path) if failure_report_path else DATA_DIR / "eval" / doc_id / "failure_report.json"
    if not report_path.exists():
        raise FileNotFoundError(f"Not found: {report_path}")

    report = read_json(report_path)
    queue: Dict[int, Dict[str, Any]] = {}

    for item in report.get("coverage_missing", []):
        query = str(item.get("query") or "")
        for page_no in item.get("suggest_refine_pages") or item.get("missing_pages") or []:
            add_page(queue, int(page_no), "eval_coverage_missing", query)

    for case in report.get("failed_cases", []):
        if case.get("failure_reason") != "coverage_missing":
            continue
        query = str(case.get("query") or "")
        for diagnostic in case.get("diagnostics", []):
            if diagnostic.get("type") != "coverage_missing":
                continue
            for page_no in diagnostic.get("suggest_refine_pages") or diagnostic.get("missing_pages") or []:
                add_page(queue, int(page_no), "eval_coverage_missing", query)

    pages_text_report_path = (
        Path(missing_coverage_path)
        if missing_coverage_path
        else DATA_DIR / "refine" / doc_id / "missing_coverage.json"
    )
    pages_text_added = add_missing_coverage_pages(queue, pages_text_report_path, min_pages_text_score)

    output = {
        "doc_id": doc_id,
        "source_failure_report": str(report_path),
        "source_missing_coverage_report": str(pages_text_report_path) if pages_text_report_path.exists() else None,
        "min_pages_text_score": min_pages_text_score,
        "pages_text_added": pages_text_added,
        "pages": sorted(queue.values(), key=lambda row: row["page_no"]),
    }
    out_path = ensure_dir(DATA_DIR / "refine" / doc_id) / "refine_queue.json"
    write_json(out_path, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Build refine_queue.json from eval failure_report.json.")
    parser.add_argument("doc_id")
    parser.add_argument("failure_report_json", nargs="?")
    parser.add_argument("--missing-coverage-json")
    parser.add_argument("--min-pages-text-score", type=float, default=15.0)
    args = parser.parse_args()

    queue = build_queue(
        args.doc_id,
        args.failure_report_json,
        args.missing_coverage_json,
        args.min_pages_text_score,
    )
    print(f"[OK] pages: {len(queue['pages'])}")
    print(f"[OK] pages_text_added: {queue['pages_text_added']}")
    print(f"[OK] output: {DATA_DIR / 'refine' / args.doc_id / 'refine_queue.json'}")
    for row in queue["pages"]:
        print(f"page={row['page_no']} reason={','.join(row['reason'])} queries={row['queries']}")


if __name__ == "__main__":
    main()
