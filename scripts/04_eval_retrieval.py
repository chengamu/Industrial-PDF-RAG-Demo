import argparse
from pathlib import Path
from typing import Dict, List

from common import DATA_DIR, canonicalize_text, ensure_dir, normalize_alarm_code, read_jsonl, write_json
from search_loader import load_search_engine


RECALL_KS = (1, 5, 10)


def normalize_codes(codes: List[str]) -> List[str]:
    return sorted(set(normalize_alarm_code(code) for code in codes if code))


def block_text(block: Dict) -> str:
    return canonicalize_text(
        "\n".join(
            [
                str(block.get("title") or ""),
                " ".join(block.get("alarm_codes") or []),
                str(block.get("section_path") or ""),
                str(block.get("searchable_text") or ""),
                str(block.get("raw_text") or ""),
            ]
        )
    )


def expected_codes_hit(blocks: List[Dict], expected_codes: List[str]) -> bool:
    if not expected_codes:
        return False
    expected_set = set(expected_codes)
    for block in blocks:
        block_codes = set(normalize_codes(block.get("alarm_codes") or []))
        if block_codes.intersection(expected_set):
            return True
        text = block_text(block)
        if any(code in text for code in expected_codes):
            return True
    return False


def expected_pages_hit(blocks: List[Dict], expected_pages: List[int]) -> bool:
    return bool(expected_pages and set(block.get("page_no") for block in blocks).intersection(set(expected_pages)))


def expected_keywords_hit(blocks: List[Dict], expected_keywords: List[str]) -> bool:
    if not expected_keywords:
        return False
    joined = "\n".join(block_text(block) for block in blocks)
    return all(canonicalize_text(keyword) in joined for keyword in expected_keywords)


def block_page_counts(blocks: List[Dict]) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for block in blocks:
        page_no = block.get("page_no")
        if isinstance(page_no, int) and page_no > 0:
            counts[page_no] = counts.get(page_no, 0) + 1
    return counts


def coverage_missing_diagnostic(all_blocks: List[Dict], case: Dict) -> Dict | None:
    expected_pages = [int(page) for page in (case.get("expected_pages") or [])]
    if not expected_pages:
        return None

    counts = block_page_counts(all_blocks)
    missing_pages = [page for page in expected_pages if counts.get(page, 0) == 0]
    if not missing_pages:
        return None
    return {
        "type": "coverage_missing",
        "query": str(case.get("query") or ""),
        "expected_pages": expected_pages,
        "missing_pages": missing_pages,
        "suggest_refine_pages": missing_pages,
    }


def hit_at_k(blocks: List[Dict], case: Dict, k: int) -> bool:
    top_blocks = blocks[:k]
    expected_codes = normalize_codes(case.get("expected_alarm_codes") or [])
    expected_pages = [int(page) for page in (case.get("expected_pages") or [])]
    expected_keywords = [str(keyword) for keyword in (case.get("expected_keywords") or []) if str(keyword)]

    checks = []
    if expected_codes:
        checks.append(expected_codes_hit(top_blocks, expected_codes))
    if expected_pages:
        checks.append(expected_pages_hit(top_blocks, expected_pages))
    if expected_keywords:
        checks.append(expected_keywords_hit(top_blocks, expected_keywords))
    return any(checks) if checks else False


def failure_reason(blocks: List[Dict], case: Dict, diagnostics: List[Dict]) -> str:
    if not blocks:
        return "no_results"
    if any(item.get("type") == "coverage_missing" for item in diagnostics):
        return "coverage_missing"

    expected_codes = normalize_codes(case.get("expected_alarm_codes") or [])
    expected_pages = [int(page) for page in (case.get("expected_pages") or [])]
    expected_keywords = [str(keyword) for keyword in (case.get("expected_keywords") or []) if str(keyword)]

    if expected_codes and not expected_codes_hit(blocks, expected_codes):
        return "exact_match_failed"
    if expected_pages and not expected_pages_hit(blocks, expected_pages):
        return "page_miss"
    if expected_keywords and not expected_keywords_hit(blocks, expected_keywords):
        return "keyword_miss"
    return "unknown_miss"


def summarize_block(block: Dict) -> Dict:
    return {
        "block_id": block.get("block_id"),
        "block_type": block.get("block_type"),
        "page_no": block.get("page_no"),
        "section_path": block.get("section_path"),
        "alarm_codes": block.get("alarm_codes"),
        "image_paths": block.get("image_paths"),
        "needs_refine": block.get("needs_refine"),
        "refine_reason": block.get("refine_reason"),
        "parse_level": block.get("parse_level"),
        "text_preview": str(block.get("raw_text") or "")[:240],
    }


def evaluate_case(engine, case: Dict, retriever: str) -> Dict:
    query = str(case.get("query") or "")
    top_k = max(int(case.get("top_k") or 10), max(RECALL_KS))
    results = engine.search(query, top_k=top_k, retriever=retriever)
    blocks = [block for _, block in results]
    diagnostics = []
    coverage_diagnostic = coverage_missing_diagnostic(engine.blocks, case)
    if coverage_diagnostic:
        diagnostics.append(coverage_diagnostic)
    recall = {f"recall@{k}": hit_at_k(blocks, case, k) for k in RECALL_KS}
    passed = recall["recall@10"]

    return {
        "query": query,
        "passed": passed,
        "retriever": retriever,
        "expected_alarm_codes": normalize_codes(case.get("expected_alarm_codes") or []),
        "expected_pages": [int(page) for page in (case.get("expected_pages") or [])],
        "expected_keywords": [str(keyword) for keyword in (case.get("expected_keywords") or []) if str(keyword)],
        "recall": recall,
        "diagnostics": diagnostics,
        "failure_reason": None if passed else failure_reason(blocks, case, diagnostics),
        "top1": summarize_block(blocks[0]) if blocks else None,
        "top10": [summarize_block(block) for block in blocks[:10]],
    }


def compute_metrics(report_cases: List[Dict]) -> Dict:
    total = len(report_cases)
    if total == 0:
        return {f"recall@{k}": 0.0 for k in RECALL_KS}
    return {
        f"recall@{k}": sum(1 for item in report_cases if item["recall"][f"recall@{k}"]) / total
        for k in RECALL_KS
    }


def main(doc_id: str, cases_path: str, retriever: str) -> None:
    cases_file = Path(cases_path)
    if not cases_file.exists():
        raise FileNotFoundError(f"Not found: {cases_file}")

    engine = load_search_engine(doc_id)
    cases = read_jsonl(cases_file)
    report_cases = [evaluate_case(engine, case, retriever) for case in cases]
    failed_cases = [item for item in report_cases if not item["passed"]]
    coverage_missing = [
        diagnostic
        for item in report_cases
        for diagnostic in item.get("diagnostics", [])
        if diagnostic.get("type") == "coverage_missing"
    ]
    metrics = compute_metrics(report_cases)

    report = {
        "doc_id": doc_id,
        "retriever": retriever,
        "case_count": len(report_cases),
        "passed_count": len(report_cases) - len(failed_cases),
        "failed_count": len(failed_cases),
        "metrics": metrics,
        "coverage_missing": coverage_missing,
        "cases": report_cases,
    }
    failure_report = {
        "doc_id": doc_id,
        "retriever": retriever,
        "failed_count": len(failed_cases),
        "coverage_missing": coverage_missing,
        "failed_cases": failed_cases,
    }

    out_dir = ensure_dir(DATA_DIR / "eval" / doc_id)
    report_path = out_dir / "retrieval_report.json"
    failure_path = out_dir / "failure_report.json"
    named_report_path = out_dir / f"retrieval_report_{retriever}.json"
    named_failure_path = out_dir / f"failure_report_{retriever}.json"
    write_json(report_path, report)
    write_json(failure_path, failure_report)
    write_json(named_report_path, report)
    write_json(named_failure_path, failure_report)

    print(f"[OK] retriever: {retriever}")
    print(f"[OK] cases: {len(report_cases)}")
    print(f"[OK] passed: {len(report_cases) - len(failed_cases)}")
    print(f"[OK] failed: {len(failed_cases)}")
    print(f"[OK] coverage_missing: {len(coverage_missing)}")
    print(f"[OK] recall@1: {metrics['recall@1']:.3f}")
    print(f"[OK] recall@5: {metrics['recall@5']:.3f}")
    print(f"[OK] recall@10: {metrics['recall@10']:.3f}")
    print(f"[OK] report: {report_path}")
    print(f"[OK] failure report: {failure_path}")
    print(f"[OK] named report: {named_report_path}")
    print(f"[OK] named failure report: {named_failure_path}")
    if failed_cases:
        print("[FAILED CASES]")
        for item in failed_cases:
            print("-" * 80)
            print(f"query: {item['query']}")
            print(f"reason: {item['failure_reason']}")
            print(f"expected_alarm_codes: {item['expected_alarm_codes']}")
            print(f"expected_pages: {item['expected_pages']}")
            print(f"expected_keywords: {item['expected_keywords']}")
            print(f"top1: {item['top1']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate retrieval recall and failure cases.")
    parser.add_argument("doc_id")
    parser.add_argument("cases_jsonl")
    parser.add_argument("--retriever", choices=["bm25", "hybrid"], default="bm25")
    args = parser.parse_args()

    main(args.doc_id, args.cases_jsonl, args.retriever)
