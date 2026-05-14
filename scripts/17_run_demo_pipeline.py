import argparse
import importlib.util
from pathlib import Path
from typing import Any, Dict, List

from common import DATA_DIR, load_dotenv_if_available, read_json, read_jsonl


def load_script(filename: str, module_name: str):
    script_path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_build_blocks(build_module, doc_id: str) -> Path:
    layout_json_path = DATA_DIR / "parsed" / doc_id / "layout.json"
    blocks = build_module.build_blocks(doc_id, layout_json_path)
    out_dir = DATA_DIR / "blocks" / doc_id
    draft_path = out_dir / "draft_blocks.jsonl"
    final_path = out_dir / "final_blocks.jsonl"
    compat_path = out_dir / "blocks.jsonl"
    build_module.write_jsonl(draft_path, blocks)
    build_module.write_jsonl(final_path, blocks)
    build_module.write_jsonl(compat_path, blocks)
    return final_path


def is_up_to_date(target: Path, sources: List[Path]) -> bool:
    if not target.exists():
        return False
    target_mtime = target.stat().st_mtime
    for source in sources:
        if source.exists() and source.stat().st_mtime > target_mtime:
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PDF -> Layout -> Retrieval Blocks -> OCR Fix -> KB -> KG demo pipeline.")
    parser.add_argument("pdf_path")
    parser.add_argument("--doc-id", default="")
    parser.add_argument("--skip-parse", action="store_true")
    parser.add_argument("--skip-ocr", action="store_true")
    parser.add_argument("--garbled-top", type=int, default=20)
    parser.add_argument("--garbled-min-score", type=float, default=2.0)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--retrievers", nargs="+", default=["bm25"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    load_dotenv_if_available()
    parse_module = load_script("01_parse_pdf.py", "parse_pdf_script")
    build_module = load_script("02_build_blocks.py", "build_blocks_script")
    merge_module = load_script("09_merge_blocks.py", "merge_blocks_script")
    detect_module = load_script("14_detect_garbled_pages.py", "detect_garbled_pages_script")
    ocr_module = load_script("15_ocr_pages_dashscope.py", "ocr_pages_dashscope_script")
    fix_module = load_script("16_fix_blocks_with_ocr.py", "fix_blocks_with_ocr_script")
    kb_module = load_script("12_build_kb_blocks.py", "build_kb_blocks_script")
    graph_module = load_script("13_build_kb_graph.py", "build_kb_graph_script")
    validate_module = load_script("11_validate_rag_pipeline.py", "validate_rag_pipeline_script")

    pdf_path = str(Path(args.pdf_path).resolve())
    doc_id = args.doc_id.strip()
    if not args.skip_parse:
        doc_id = parse_module.parse_pdf(pdf_path, doc_id or None)
    if not doc_id:
        doc_id = parse_module.slugify_doc_id(Path(pdf_path).stem)

    parsed_layout_path = DATA_DIR / "parsed" / doc_id / "layout.json"
    final_blocks_path = DATA_DIR / "blocks" / doc_id / "final_blocks.jsonl"
    if args.force or not is_up_to_date(final_blocks_path, [parsed_layout_path]):
        final_blocks_path = run_build_blocks(build_module, doc_id)
    refined_blocks_path = DATA_DIR / "refine" / doc_id / "refined_blocks.jsonl"
    if refined_blocks_path.exists():
        if args.force or not is_up_to_date(final_blocks_path, [DATA_DIR / "blocks" / doc_id / "draft_blocks.jsonl", refined_blocks_path]):
            merge_summary = merge_module.merge_blocks(doc_id)
            final_blocks_path = Path(merge_summary["final_path"])
            print(f"[PIPELINE] merged refined blocks -> {final_blocks_path}")
    print(f"[PIPELINE] final_blocks: {final_blocks_path}")

    report, garbled_report_path = detect_module.detect_garbled_pages(
        final_blocks_path,
        min_score=args.garbled_min_score,
        top=args.garbled_top,
    )
    print(f"[PIPELINE] garbled pages: {report['garbled_page_count']}")

    final_blocks_for_downstream = final_blocks_path
    if not args.skip_ocr and report["pages"]:
        fixed_path = DATA_DIR / "blocks" / doc_id / "final_blocks_ocr_fixed.jsonl"
        ocr_pages_path = DATA_DIR / "ocr" / doc_id / "ocr_pages.jsonl"
        if args.force or not is_up_to_date(ocr_pages_path, [DATA_DIR / "ocr" / doc_id / "garbled_pages.json"]):
            ocr_pages, ocr_pages_path = ocr_module.ocr_pages(doc_id, report["pages"], force=args.force)
            print(f"[PIPELINE] ocr_pages: {len(ocr_pages)} -> {ocr_pages_path}")
        if args.force or not is_up_to_date(fixed_path, [final_blocks_path, ocr_pages_path]):
            _, fixed_path = fix_module.fix_blocks_with_ocr(final_blocks_path, ocr_pages_path)
        final_blocks_for_downstream = fixed_path
        print(f"[PIPELINE] final_blocks_ocr_fixed: {fixed_path}")

    kb_path = DATA_DIR / "kb" / doc_id / "kb_blocks.jsonl"
    if args.force or not is_up_to_date(kb_path, [Path(final_blocks_for_downstream)]):
        kb_rows, kb_path = kb_module.build_kb_blocks(final_blocks_for_downstream)
    else:
        kb_rows = read_jsonl(kb_path)
    print(f"[PIPELINE] kb_blocks: {len(kb_rows)} -> {kb_path}")

    graph_path = DATA_DIR / "kb" / doc_id / "kb_graph.json"
    if args.force or not is_up_to_date(graph_path, [kb_path]):
        graph, graph_path = graph_module.build_graph(kb_path)
    else:
        graph = read_json(graph_path)
    print(f"[PIPELINE] kb_graph: nodes={graph['node_count']} edges={graph['edge_count']} -> {graph_path}")

    if args.validate:
        queries_path = DATA_DIR / "eval" / doc_id / "eval_queries.jsonl"
        if queries_path.exists():
            validation_args = type(
                "ValidationArgs",
                (),
                {
                    "final_blocks_jsonl": str(final_blocks_for_downstream),
                    "queries_jsonl": str(queries_path),
                    "retrievers": args.retrievers,
                    "embedding_model": validate_module.DEFAULT_EMBEDDING_MODEL,
                    "rerank_model": validate_module.DEFAULT_RERANK_MODEL,
                    "kb_blocks_jsonl": str(kb_path),
                    "kb_graph_json": str(graph_path),
                    "out": "",
                },
            )()
            queries = read_jsonl(queries_path)
            retriever_reports: Dict[str, Any] = {}
            pipeline = None
            kb_retriever = None
            if any(retriever in {"bm25", "vector", "hybrid", "hybrid_rerank"} for retriever in validation_args.retrievers):
                pipeline = validate_module.load_pipeline(
                    validation_args.final_blocks_jsonl,
                    embedding_model=validation_args.embedding_model,
                    rerank_model=validation_args.rerank_model,
                )
                node_validation = pipeline.validate_nodes()
                print(f"[PIPELINE] validation text_nodes={node_validation['text_node_count']}")
            else:
                node_validation = None
            if any(retriever in {"kb_bm25", "kb_hybrid", "graph_expanded", "graph_expanded_rerank"} for retriever in validation_args.retrievers):
                kb_retriever = validate_module.load_kb_graph_retriever(
                    validation_args.kb_blocks_jsonl,
                    validation_args.kb_graph_json,
                    embedding_model=validation_args.embedding_model,
                    rerank_model=validation_args.rerank_model,
                )
            for retriever in validation_args.retrievers:
                active = kb_retriever if retriever in {"kb_bm25", "kb_hybrid", "graph_expanded", "graph_expanded_rerank"} else pipeline
                case_results = [validate_module.evaluate_case(active, case, retriever) for case in queries]
                retriever_reports[retriever] = {"summary": validate_module.summarize_cases(case_results), "cases": case_results}
            for retriever, report_row in retriever_reports.items():
                summary = report_row["summary"]
                print(f"[PIPELINE] {retriever}: overall_hit_rate={summary['overall_hit_rate']:.3f} avg_top1_score={summary['avg_top1_score']:.4f}")


if __name__ == "__main__":
    main()
