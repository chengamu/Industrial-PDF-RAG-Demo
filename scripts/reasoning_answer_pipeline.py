import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Tuple

from openai import OpenAI

from common import canonicalize_text, extract_alarm_codes, load_dotenv_if_available, read_json


RULE_KEYWORDS = {
    "safety_warning": ["安全", "警告", "注意", "禁止", "断电", "高压", "触电"],
    "inspection_step": ["检查", "确认", "步骤", "更换", "设置", "设定", "复位", "查看", "画面", "测量"],
    "parameter": ["参数", "设定", "设置值", "PWE", "NO.", "#", "PMC ADDRESS", "PLC"],
    "possible_cause": ["原因", "可能", "异常", "故障", "错误", "通信", "报警"],
}

ANSWER_TEMPLATES = {
    "alarm_troubleshooting": [
        "报警结论",
        "报警优先级",
        "每个报警的手册出处",
        "可能原因",
        "现场检查步骤",
        "禁止操作/安全提醒",
        "需要客户补充的信息",
    ],
    "alarm_explain": [
        "报警结论",
        "每个报警的手册出处",
        "可能原因",
        "禁止操作/安全提醒",
        "需要客户补充的信息",
    ],
    "procedure_answer": [
        "操作目标",
        "手册出处",
        "现场检查步骤",
        "禁止操作/安全提醒",
        "需要客户补充的信息",
    ],
    "parameter_answer": [
        "参数结论",
        "手册出处",
        "参数说明",
        "现场检查步骤",
        "禁止操作/安全提醒",
        "需要客户补充的信息",
    ],
    "safety_answer": [
        "安全结论",
        "手册出处",
        "禁止操作/安全提醒",
        "需要客户补充的信息",
    ],
    "general_answer": [
        "问题结论",
        "手册出处",
        "可能原因",
        "现场检查步骤",
        "禁止操作/安全提醒",
        "需要客户补充的信息",
    ],
}


def unique_keep_order(items: Iterable[str]) -> List[str]:
    seen = set()
    output: List[str] = []
    for item in items:
        normalized = str(item).strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            output.append(normalized)
    return output


def evidence_text(item: Dict[str, Any]) -> str:
    return canonicalize_text(
        "\n".join(
            [
                str(item.get("text") or ""),
                str(item.get("section_path") or ""),
                " ".join(str(code) for code in (item.get("alarm_codes") or [])),
                " ".join(str(entity) for entity in (item.get("entities") or [])),
            ]
        )
    )


def infer_kb_type(item: Dict[str, Any]) -> str:
    kb_type = str(item.get("kb_type") or "").strip()
    if kb_type:
        return kb_type
    if item.get("alarm_codes"):
        return "alarm"
    text = evidence_text(item)
    if any(keyword in text for keyword in RULE_KEYWORDS["safety_warning"]):
        return "warning"
    if any(keyword in text for keyword in RULE_KEYWORDS["inspection_step"]):
        return "procedure"
    return "text"


def classify_evidence_item(item: Dict[str, Any], parsed_query: Dict[str, Any]) -> str:
    text = evidence_text(item)
    kb_type = infer_kb_type(item)
    query_codes = {str(code).upper() for code in parsed_query.get("alarm_codes") or []}
    item_codes = {str(code).upper() for code in item.get("alarm_codes") or []}

    if kb_type == "warning" or any(keyword in text for keyword in RULE_KEYWORDS["safety_warning"]):
        return "safety_warning"
    if query_codes and item_codes.intersection(query_codes):
        return "alarm_definition"
    if kb_type == "alarm":
        return "alarm_definition"
    if kb_type == "procedure" or any(keyword in text for keyword in RULE_KEYWORDS["inspection_step"]):
        return "inspection_step"
    if any(keyword in text for keyword in RULE_KEYWORDS["parameter"]):
        return "parameter"
    if any(keyword in text for keyword in RULE_KEYWORDS["possible_cause"]):
        return "possible_cause"
    return "context"


def answer_type_from_query(parsed_query: Dict[str, Any]) -> str:
    query_type = str(parsed_query.get("query_type") or "")
    if query_type == "alarm_diagnosis":
        return "alarm_troubleshooting"
    if query_type == "alarm_explain":
        return "alarm_explain"
    if query_type == "procedure_lookup":
        return "procedure_answer"
    if query_type == "parameter_lookup":
        return "parameter_answer"
    if query_type == "safety_lookup":
        return "safety_answer"
    return "general_answer"


def evidence_summary(item: Dict[str, Any], role: str, why: str) -> Dict[str, Any]:
    return {
        "kb_id": item.get("kb_id"),
        "block_id": item.get("block_id"),
        "role": role,
        "page_no": item.get("page_no"),
        "section_path": item.get("section_path"),
        "why": why,
        "text": item.get("text"),
        "image_paths": item.get("image_paths") or [],
        "sources": item.get("sources") or [],
        "alarm_codes": item.get("alarm_codes") or [],
        "entities": item.get("entities") or [],
    }


def build_evidence_groups(evidence_package: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    parsed_query = evidence_package["parsed_query"]
    groups = {
        "alarm_definition": [],
        "possible_cause": [],
        "inspection_step": [],
        "parameter": [],
        "safety_warning": [],
        "context": [],
        "irrelevant": [],
    }
    for item in evidence_package.get("topk") or []:
        enriched = dict(item)
        enriched["kb_type"] = infer_kb_type(enriched)
        role = classify_evidence_item(enriched, parsed_query)
        groups.setdefault(role, []).append(enriched)
    return groups


def choose_main_evidence(evidence_package: Dict[str, Any], groups: Dict[str, List[Dict[str, Any]]], answer_type: str) -> List[Dict[str, Any]]:
    parsed_query = evidence_package["parsed_query"]
    query_codes = {str(code).upper() for code in parsed_query.get("alarm_codes") or []}
    ranked = evidence_package.get("topk") or []

    def sort_key(item: Dict[str, Any]) -> Tuple[float, float, float, float]:
        item_codes = {str(code).upper() for code in item.get("alarm_codes") or []}
        role = classify_evidence_item(item | {"kb_type": infer_kb_type(item)}, parsed_query)
        exact_alarm = 1.0 if query_codes and item_codes.intersection(query_codes) else 0.0
        role_priority = {
            "alarm_definition": 1.0,
            "possible_cause": 0.8,
            "inspection_step": 0.7,
            "parameter": 0.6,
            "safety_warning": 0.5,
            "context": 0.2,
        }.get(role, 0.0)
        procedure_boost = 1.0 if answer_type == "procedure_answer" and infer_kb_type(item) == "procedure" else 0.0
        score = float(item.get("adjusted_rerank_score") or item.get("rerank_score") or 0.0)
        return (exact_alarm, procedure_boost, role_priority, score)

    sorted_items = sorted(ranked, key=sort_key, reverse=True)
    selected: List[Dict[str, Any]] = []
    for item in sorted_items:
        role = classify_evidence_item(item | {"kb_type": infer_kb_type(item)}, parsed_query)
        why = "高相关主证据"
        if query_codes and set(str(code).upper() for code in item.get("alarm_codes") or []).intersection(query_codes):
            why = f"命中报警码 {next(iter(query_codes))}"
        elif role == "inspection_step":
            why = "命中现场检查或操作步骤"
        elif role == "safety_warning":
            why = "命中维修安全提醒"
        selected.append(evidence_summary(item, role, why))
        if len(selected) >= 3:
            break
    return selected[:3]


def choose_supporting_evidence(
    evidence_package: Dict[str, Any],
    groups: Dict[str, List[Dict[str, Any]]],
    main_evidence: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    main_ids = {item["kb_id"] for item in main_evidence}
    supporting: List[Dict[str, Any]] = []
    safety: List[Dict[str, Any]] = []
    for role in ["inspection_step", "possible_cause", "parameter", "context", "alarm_definition"]:
        for item in groups.get(role, []):
            if item.get("kb_id") in main_ids:
                continue
            why = "Graph 扩展得到的辅助证据" if "graph" in (item.get("sources") or []) else "辅助证据"
            supporting.append(evidence_summary(item, role, why))
            if len(supporting) >= 6:
                break
        if len(supporting) >= 6:
            break
    for item in groups.get("safety_warning", []):
        if item.get("kb_id") in main_ids:
            continue
        safety.append(evidence_summary(item, "safety_warning", "维修或更换相关安全提醒"))
        if len(safety) >= 2:
            break
    return supporting[:6], safety[:2]


def default_missing_info(answer_type: str) -> List[str]:
    mapping = {
        "alarm_troubleshooting": [
            "请提供 CNC 当前完整报警画面",
            "请确认是否同时存在其他 SP/SV/DS/FSSB 报警",
            "请提供主轴/伺服放大器 LED 显示状态",
            "请确认报警发生时机",
            "请确认近期是否更换过电机、放大器、电缆或参数",
        ],
        "procedure_answer": [
            "请确认当前所处画面或参数页面",
            "请确认机床是否已经停机并处于可操作状态",
            "请确认当前控制单元/板卡型号",
        ],
        "parameter_answer": [
            "请提供参数号和当前参数值",
            "请确认是否允许参数写入",
            "请说明目标功能或想达到的设定结果",
        ],
        "safety_answer": [
            "请说明当前维修对象和操作步骤",
            "请确认是否涉及断电、高压或板卡拆装",
        ],
        "general_answer": [
            "请提供更完整的现场现象或报警文本",
            "请提供相关画面或手册截图",
        ],
    }
    return mapping.get(answer_type, mapping["general_answer"])


def can_answer_check(parsed_query: Dict[str, Any], groups: Dict[str, List[Dict[str, Any]]], answer_type: str) -> Tuple[bool, str, List[str]]:
    query_codes = {str(code).upper() for code in parsed_query.get("alarm_codes") or []}
    if query_codes:
        for item in groups.get("alarm_definition", []):
            item_codes = {str(code).upper() for code in item.get("alarm_codes") or []}
            if item_codes.intersection(query_codes):
                return True, "", default_missing_info(answer_type)
        return False, "缺少命中同报警码的手册证据", default_missing_info(answer_type)

    non_empty_groups = sum(1 for key in ["alarm_definition", "possible_cause", "inspection_step", "parameter", "safety_warning"] if groups.get(key))
    if groups.get("inspection_step") and non_empty_groups >= 2:
        return True, "", default_missing_info(answer_type)
    if groups.get("alarm_definition") or groups.get("possible_cause") or groups.get("inspection_step"):
        return True, "", default_missing_info(answer_type)
    return False, "手册证据不足，缺少明确步骤或故障定义", default_missing_info(answer_type)


def confidence_score(evidence_package: Dict[str, Any], can_answer: bool, groups: Dict[str, List[Dict[str, Any]]]) -> float:
    topk = evidence_package.get("topk") or []
    top_score = float((topk[0].get("adjusted_rerank_score") or topk[0].get("rerank_score") or 0.0) if topk else 0.0)
    coverage = sum(1 for key in ["alarm_definition", "possible_cause", "inspection_step", "parameter", "safety_warning"] if groups.get(key))
    score = min(0.98, max(0.15, 0.45 + min(top_score, 1.0) * 0.35 + coverage * 0.05))
    if not can_answer:
        score = min(score, 0.45)
    return round(score, 2)


def build_reasoning(evidence_package: Dict[str, Any]) -> Dict[str, Any]:
    parsed_query = evidence_package["parsed_query"]
    answer_type = answer_type_from_query(parsed_query)
    groups = build_evidence_groups(evidence_package)
    can_answer, reason, missing_info = can_answer_check(parsed_query, groups, answer_type)
    main_evidence = choose_main_evidence(evidence_package, groups, answer_type)
    supporting_evidence, safety_evidence = choose_supporting_evidence(evidence_package, groups, main_evidence)
    confidence = confidence_score(evidence_package, can_answer, groups)

    grouped_summary = {
        role: [evidence_summary(item, role, f"{role} 证据") for item in items[:4]]
        for role, items in groups.items()
        if items and role != "irrelevant"
    }
    sections = ANSWER_TEMPLATES[answer_type]
    return {
        "query": evidence_package["query"],
        "parsed_query": parsed_query,
        "can_answer": can_answer,
        "reason": reason,
        "answer_type": answer_type,
        "confidence": confidence,
        "evidence_groups": grouped_summary,
        "main_evidence": main_evidence[:3],
        "supporting_evidence": supporting_evidence[:6],
        "safety_evidence": safety_evidence[:2],
        "missing_info": missing_info,
        "answer_plan": {
            "template": answer_type,
            "sections": sections,
        },
    }


def evidence_material(reasoning: Dict[str, Any]) -> str:
    blocks: List[str] = []
    for bucket_name, items in [
        ("主证据", reasoning.get("main_evidence") or []),
        ("辅助证据", reasoning.get("supporting_evidence") or []),
        ("安全证据", reasoning.get("safety_evidence") or []),
    ]:
        for index, item in enumerate(items, start=1):
            blocks.append(
                "\n".join(
                    [
                        f"[{bucket_name}{index}]",
                        f"类型: {item.get('role')}",
                        f"页码: {item.get('page_no')}",
                        f"章节: {item.get('section_path')}",
                        f"KB: {item.get('kb_id')}",
                        f"内容: {item.get('text')}",
                    ]
                )
            )
    return "\n\n".join(blocks)


def build_answer_prompt(evidence_package: Dict[str, Any], reasoning: Dict[str, Any]) -> List[Dict[str, str]]:
    parsed_query = reasoning["parsed_query"]
    system_prompt = (
        "你是工业 CNC 维修手册问答助手。"
        "你只能根据证据材料回答，不允许编造。"
        "如果证据不足，请明确说明，并列出需要客户补充的信息。"
        "涉及维修、更换、断电、高压、放大器、电池、保险丝时，必须给出安全提醒。"
        "输出必须是 JSON 对象，字段为 answer、citations、confidence。"
    )
    user_prompt = "\n\n".join(
        [
            f"【用户问题】\n{evidence_package['query']}",
            "【解析结果】\n"
            + json.dumps(
                {
                    "alarm_codes": parsed_query.get("alarm_codes"),
                    "entities": parsed_query.get("entities"),
                    "intent": parsed_query.get("intent"),
                    "query_type": parsed_query.get("query_type"),
                },
                ensure_ascii=False,
                indent=2,
            ),
            "【Reasoning 结果】\n"
            + json.dumps(
                {
                    "can_answer": reasoning.get("can_answer"),
                    "answer_type": reasoning.get("answer_type"),
                    "confidence": reasoning.get("confidence"),
                    "missing_info": reasoning.get("missing_info"),
                    "answer_plan": reasoning.get("answer_plan"),
                },
                ensure_ascii=False,
                indent=2,
            ),
            f"【证据材料】\n{evidence_material(reasoning)}",
            "【输出要求】\n"
            "1. answer 下必须包含模板要求的所有 section。\n"
            "2. 每条关键结论必须尽量引用页码和章节。\n"
            "3. 不允许出现证据中没有的报警码或页码。\n"
            "4. 如果 can_answer=false，必须明确说明证据不足。\n"
            "5. 输出 JSON，不要额外解释。",
            "【输出模板】\n"
            + json.dumps(
                {
                    "answer": {section: "" for section in reasoning["answer_plan"]["sections"]},
                    "citations": [{"page_no": 0, "kb_id": ""}],
                    "confidence": reasoning["confidence"],
                },
                ensure_ascii=False,
                indent=2,
            ),
        ]
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        raise ValueError("No JSON object found in model output.")
    return json.loads(match.group(0))


def rule_based_answer(reasoning: Dict[str, Any]) -> Dict[str, Any]:
    sections = reasoning["answer_plan"]["sections"]
    main = reasoning.get("main_evidence") or []
    supporting = reasoning.get("supporting_evidence") or []
    safety = reasoning.get("safety_evidence") or []

    def collect_text(items: List[Dict[str, Any]], limit: int = 3) -> List[str]:
        output = []
        for item in items[:limit]:
            text = str(item.get("text") or "").strip()
            if text:
                output.append(text[:160])
        return output

    answer = {section: "" for section in sections}
    if "报警结论" in answer:
        answer["报警结论"] = "；".join(collect_text(main, 2)) or "手册证据不足，无法确认具体报警结论。"
    if "报警优先级" in answer:
        answer["报警优先级"] = "建议先停止继续加工，优先确认报警与通信链路相关证据。"
    if "每个报警的手册出处" in answer:
        answer["每个报警的手册出处"] = [
            {
                "page_no": item.get("page_no"),
                "section_path": item.get("section_path"),
                "kb_id": item.get("kb_id"),
            }
            for item in main[:3]
        ]
    if "手册出处" in answer:
        answer["手册出处"] = [
            {
                "page_no": item.get("page_no"),
                "section_path": item.get("section_path"),
                "kb_id": item.get("kb_id"),
            }
            for item in main[:3]
        ]
    if "可能原因" in answer:
        answer["可能原因"] = collect_text(
            [item for item in supporting if item.get("role") in {"possible_cause", "alarm_definition", "context"}],
            4,
        ) or ["手册未给出明确原因，请结合现场信息补充判断。"]
    if "现场检查步骤" in answer:
        answer["现场检查步骤"] = collect_text(
            [item for item in main + supporting if item.get("role") == "inspection_step"],
            5,
        ) or ["手册证据中未检出明确步骤，请补充现场画面和当前状态。"]
    if "参数说明" in answer:
        answer["参数说明"] = collect_text(
            [item for item in main + supporting if item.get("role") == "parameter"],
            4,
        ) or ["手册证据中未检出明确参数说明。"]
    if "安全结论" in answer:
        answer["安全结论"] = "维修前应先确认风险点，并严格按手册安全注意事项操作。"
    if "操作目标" in answer:
        answer["操作目标"] = "根据手册相关步骤完成当前操作/查看动作。"
    if "问题结论" in answer:
        answer["问题结论"] = "已根据当前证据整理相关手册内容，建议按步骤继续确认。"
    if "禁止操作/安全提醒" in answer:
        answer["禁止操作/安全提醒"] = collect_text(safety, 4) or [
            "涉及放大器、板卡、电池、电缆插拔前，应先断开外部供给电源。",
            "未确认故障原因前，不建议继续加工或带负载运行。",
        ]
    if "需要客户补充的信息" in answer:
        answer["需要客户补充的信息"] = reasoning.get("missing_info") or []

    citations = [
        {"page_no": item.get("page_no"), "kb_id": item.get("kb_id")}
        for item in (main + supporting + safety)[:8]
    ]
    return {"answer": answer, "citations": citations, "confidence": reasoning.get("confidence", 0.5)}


def generate_llm_answer(
    evidence_package: Dict[str, Any],
    reasoning: Dict[str, Any],
    stream_handler: Callable[[str], None] | None = None,
) -> Tuple[Dict[str, Any], str]:
    load_dotenv_if_available()
    model = os.getenv("OPENAI_MODEL", "").strip()
    base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not (model and base_url and api_key):
        raise RuntimeError("Missing OPENAI_MODEL / OPENAI_BASE_URL / OPENAI_API_KEY in .env")

    client = OpenAI(api_key=api_key, base_url=base_url)
    messages = build_answer_prompt(evidence_package, reasoning)
    if stream_handler is not None:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.1,
            stream=True,
        )
        parts: List[str] = []
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if not delta:
                continue
            parts.append(delta)
            stream_handler(delta)
        content = "".join(parts)
    else:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.1,
        )
        content = response.choices[0].message.content or ""
    return extract_json_object(content), content


def validate_answer(answer_obj: Dict[str, Any], reasoning: Dict[str, Any]) -> Dict[str, Any]:
    sections = set(reasoning["answer_plan"]["sections"])
    answer_sections = set((answer_obj.get("answer") or {}).keys())
    evidence_pages = {
        int(item.get("page_no"))
        for item in reasoning.get("main_evidence", []) + reasoning.get("supporting_evidence", []) + reasoning.get("safety_evidence", [])
        if isinstance(item.get("page_no"), int)
    }
    evidence_kb_ids = {
        str(item.get("kb_id"))
        for item in reasoning.get("main_evidence", []) + reasoning.get("supporting_evidence", []) + reasoning.get("safety_evidence", [])
        if str(item.get("kb_id") or "").strip()
    }
    citations = answer_obj.get("citations") or []
    citation_pages = {int(item.get("page_no")) for item in citations if isinstance(item.get("page_no"), int)}
    citation_kb_ids = {str(item.get("kb_id")) for item in citations if str(item.get("kb_id") or "").strip()}
    answer_text = canonicalize_text(json.dumps(answer_obj.get("answer") or {}, ensure_ascii=False))
    answer_codes = set(extract_alarm_codes(answer_text))
    evidence_codes = {
        str(code).upper()
        for item in reasoning.get("main_evidence", []) + reasoning.get("supporting_evidence", []) + reasoning.get("safety_evidence", [])
        for code in (item.get("alarm_codes") or [])
    } | {str(code).upper() for code in reasoning.get("parsed_query", {}).get("alarm_codes") or []}

    checks = {
        "has_required_sections": sections.issubset(answer_sections),
        "citations_page_in_evidence": citation_pages.issubset(evidence_pages),
        "citations_kb_in_evidence": citation_kb_ids.issubset(evidence_kb_ids),
        "safety_section_present": (
            "禁止操作/安全提醒" in answer_sections and bool((answer_obj.get("answer") or {}).get("禁止操作/安全提醒"))
        )
        if reasoning.get("parsed_query", {}).get("need_safety") or reasoning.get("safety_evidence")
        else True,
        "insufficient_case_explained": (
            "证据不足" in answer_text or bool((answer_obj.get("answer") or {}).get("需要客户补充的信息"))
        )
        if not reasoning.get("can_answer")
        else True,
        "no_unseen_alarm_codes": answer_codes.issubset(evidence_codes) if evidence_codes else True,
    }
    checks["ok"] = all(checks.values())
    return checks


def build_final_answer(
    evidence_package: Dict[str, Any],
    reasoning: Dict[str, Any],
    stream_handler: Callable[[str], None] | None = None,
) -> Dict[str, Any]:
    try:
        llm_answer, raw_content = generate_llm_answer(
            evidence_package,
            reasoning,
            stream_handler=stream_handler,
        )
        validation = validate_answer(llm_answer, reasoning)
        if validation["ok"]:
            return {
                "query": evidence_package["query"],
                "reasoning": reasoning,
                "answer": llm_answer,
                "validation": validation,
                "generation_mode": "llm",
            }
    except Exception as exc:
        raw_content = f"LLM generation failed: {type(exc).__name__}: {exc}"

    fallback = rule_based_answer(reasoning)
    validation = validate_answer(fallback, reasoning)
    return {
        "query": evidence_package["query"],
        "reasoning": reasoning,
        "answer": fallback,
        "validation": validation,
        "generation_mode": "template_fallback",
        "fallback_reason": raw_content,
    }


def load_reasoning_input(path_or_obj: str | Path | Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(path_or_obj, dict):
        return path_or_obj
    return read_json(path_or_obj)
