import argparse

from common import read_json, write_json
from reasoning_answer_pipeline import build_final_answer, build_reasoning, load_reasoning_input


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate grounded answer from TopK Evidence Package.")
    parser.add_argument("evidence_json")
    parser.add_argument("--reasoning-json", default="")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    evidence = load_reasoning_input(args.evidence_json)
    reasoning = read_json(args.reasoning_json) if args.reasoning_json else build_reasoning(evidence)
    output = build_final_answer(evidence, reasoning)

    if args.out:
        write_json(args.out, output)
        print(f"[OK] output: {args.out}")

    print(f"[OK] query: {output['query']}")
    print(f"[OK] generation_mode: {output['generation_mode']}")
    print(f"[OK] validation_ok: {output['validation']['ok']}")
    print(f"[OK] can_answer: {output['reasoning']['can_answer']}")
    print(f"[OK] answer_type: {output['reasoning']['answer_type']}")


if __name__ == "__main__":
    main()
