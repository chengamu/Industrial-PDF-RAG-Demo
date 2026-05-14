import argparse

from common import write_json
from reasoning_answer_pipeline import build_reasoning, load_reasoning_input


def main() -> None:
    parser = argparse.ArgumentParser(description="Build rule-based reasoning plan from TopK Evidence Package.")
    parser.add_argument("evidence_json")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    evidence = load_reasoning_input(args.evidence_json)
    reasoning = build_reasoning(evidence)

    if args.out:
        write_json(args.out, reasoning)
        print(f"[OK] output: {args.out}")

    print(f"[OK] query: {reasoning['query']}")
    print(f"[OK] can_answer: {reasoning['can_answer']}")
    print(f"[OK] answer_type: {reasoning['answer_type']}")
    print(f"[OK] confidence: {reasoning['confidence']}")
    print(f"[OK] main_evidence: {len(reasoning['main_evidence'])}")
    print(f"[OK] supporting_evidence: {len(reasoning['supporting_evidence'])}")
    print(f"[OK] safety_evidence: {len(reasoning['safety_evidence'])}")


if __name__ == "__main__":
    main()
