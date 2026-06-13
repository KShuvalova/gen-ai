from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from schema import EvalCase, JudgeResult


PROJECT_DIR = Path(__file__).parent
INPUT_DIR = PROJECT_DIR / "input"
OUTPUT_DIR = PROJECT_DIR / "output"

EVAL_CASES_PATH = INPUT_DIR / "eval_cases.json"
CLASSIFIED_PATH = OUTPUT_DIR / "classified_tickets.json"
RAG_RESULTS_PATH = OUTPUT_DIR / "rag_results.json"
TRACE_PATH = OUTPUT_DIR / "agent_trace.json"
HALLUCINATION_PATH = OUTPUT_DIR / "hallucination_report.json"


REQUIRED_TOOLS = {
    "classify_ticket_text",
    "choose_department",
    "should_escalate",
    "retrieve_policy_docs",
    "hallucination_check",
}


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: object) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def key(ticket_id: int | str) -> str:
    return str(ticket_id)


def source_hit(used_sources: list[str], expected_sources: list[str]) -> bool:
    return bool(set(used_sources) & set(expected_sources))


def all_required_tools_called(trace: dict) -> bool:
    called = {step["tool_name"] for step in trace.get("steps", [])}
    return REQUIRED_TOOLS.issubset(called)


def evaluate_one(
    case: EvalCase,
    prediction: dict,
    rag_result: dict,
    trace: dict,
    hallucination: dict,
) -> dict:
    category_ok = prediction["category"] == case.expected_category
    priority_ok = prediction["priority"] == case.expected_priority
    department_ok = prediction["department"] == case.expected_department
    escalation_ok = prediction["escalation_required"] == case.expected_escalation

    evidence_ok = bool(hallucination["evidence_quote_found"])
    source_ok = source_hit(prediction["used_sources"], case.expected_sources)

    fake_source_count = int(hallucination["fake_source_count"])
    ghost_quote_count = int(hallucination["ghost_quote_count"])

    tool_path_ok = all_required_tools_called(trace)
    tool_steps = len(trace.get("steps", []))

    judge_pass = all(
        [
            category_ok,
            priority_ok,
            department_ok,
            escalation_ok,
            evidence_ok,
            source_ok,
            tool_path_ok,
            fake_source_count == 0,
            ghost_quote_count == 0,
        ]
    )

    comment_parts = []
    if not category_ok:
        comment_parts.append(f"wrong category: predicted {prediction['category']}, expected {case.expected_category}")
    if not priority_ok:
        comment_parts.append(f"wrong priority: predicted {prediction['priority']}, expected {case.expected_priority}")
    if not department_ok:
        comment_parts.append(f"wrong department: predicted {prediction['department']}, expected {case.expected_department}")
    if not escalation_ok:
        comment_parts.append(
            f"wrong escalation: predicted {prediction['escalation_required']}, expected {case.expected_escalation}"
        )
    if not evidence_ok:
        comment_parts.append("evidence quote was not found in the original ticket")
    if not source_ok:
        comment_parts.append(
            f"no expected policy source found; used {prediction['used_sources']}, expected {case.expected_sources}"
        )
    if not tool_path_ok:
        comment_parts.append("agent did not call all required tools")
    if fake_source_count:
        comment_parts.append(f"fake sources: {fake_source_count}")
    if ghost_quote_count:
        comment_parts.append(f"ghost quotes: {ghost_quote_count}")

    if not comment_parts:
        comment_parts.append("all checks passed")

    judge = JudgeResult(
        ticket_id=case.ticket_id,
        judge_pass=judge_pass,
        category_ok=category_ok,
        priority_ok=priority_ok,
        department_ok=department_ok,
        evidence_ok=evidence_ok,
        source_ok=source_ok,
        comment="; ".join(comment_parts),
    )

    return {
        "case_id": case.id,
        "ticket_id": case.ticket_id,
        "expected_category": case.expected_category,
        "predicted_category": prediction["category"],
        "category_ok": category_ok,
        "expected_priority": case.expected_priority,
        "predicted_priority": prediction["priority"],
        "priority_ok": priority_ok,
        "expected_department": case.expected_department,
        "predicted_department": prediction["department"],
        "department_ok": department_ok,
        "expected_escalation": case.expected_escalation,
        "predicted_escalation": prediction["escalation_required"],
        "escalation_ok": escalation_ok,
        "expected_sources": case.expected_sources,
        "used_sources": prediction["used_sources"],
        "rag_source_hit": source_ok,
        "evidence_quote": prediction["evidence_quote"],
        "evidence_quote_found": evidence_ok,
        "ghost_quote_count": ghost_quote_count,
        "fake_source_count": fake_source_count,
        "tool_steps": tool_steps,
        "tool_path_ok": tool_path_ok,
        "judge_pass": judge_pass,
        "judge_comment": judge.comment,
        "rag_context_scores": rag_result.get("policy_context", []),
    }


def main() -> None:
    eval_cases_raw = load_json(EVAL_CASES_PATH)
    predictions_raw = load_json(CLASSIFIED_PATH)
    rag_raw = load_json(RAG_RESULTS_PATH)
    traces_raw = load_json(TRACE_PATH)
    hallucination_raw = load_json(HALLUCINATION_PATH)

    eval_cases = [EvalCase(**item) for item in eval_cases_raw]

    predictions_by_id = {key(item["ticket_id"]): item for item in predictions_raw}
    rag_by_id = {key(item["ticket_id"]): item for item in rag_raw}
    traces_by_id = {key(item["ticket_id"]): item for item in traces_raw}
    hallucination_by_id = {key(item["ticket_id"]): item for item in hallucination_raw}

    rows = []

    for case in eval_cases:
        ticket_key = key(case.ticket_id)

        if ticket_key not in predictions_by_id:
            raise KeyError(f"Missing prediction for ticket_id={case.ticket_id}")

        row = evaluate_one(
            case=case,
            prediction=predictions_by_id[ticket_key],
            rag_result=rag_by_id[ticket_key],
            trace=traces_by_id[ticket_key],
            hallucination=hallucination_by_id[ticket_key],
        )
        rows.append(row)

    detailed_df = pd.DataFrame(rows)

    total = len(detailed_df)

    metrics = {
        "total": int(total),
        "category_accuracy": round(float(detailed_df["category_ok"].mean()), 4),
        "priority_accuracy": round(float(detailed_df["priority_ok"].mean()), 4),
        "department_accuracy": round(float(detailed_df["department_ok"].mean()), 4),
        "escalation_accuracy": round(float(detailed_df["escalation_ok"].mean()), 4),
        "quote_valid_rate": round(float(detailed_df["evidence_quote_found"].mean()), 4),
        "rag_hit_rate": round(float(detailed_df["rag_source_hit"].mean()), 4),
        "agent_path_valid_rate": round(float(detailed_df["tool_path_ok"].mean()), 4),
        "judge_pass_rate": round(float(detailed_df["judge_pass"].mean()), 4),
        "avg_tool_steps": round(float(detailed_df["tool_steps"].mean()), 4),
        "ghost_quote_count": int(detailed_df["ghost_quote_count"].sum()),
        "fake_source_count": int(detailed_df["fake_source_count"].sum()),
    }

    eval_results_df = pd.DataFrame([metrics])
    eval_results_df.to_csv(OUTPUT_DIR / "eval_results.csv", index=False)

    detailed_df.to_csv(OUTPUT_DIR / "eval_detailed.csv", index=False)

    save_json(
        OUTPUT_DIR / "eval_detailed.json",
        detailed_df.to_dict(orient="records"),
    )

    judge_report = [
        {
            "ticket_id": row["ticket_id"],
            "judge_pass": row["judge_pass"],
            "category_ok": row["category_ok"],
            "priority_ok": row["priority_ok"],
            "department_ok": row["department_ok"],
            "evidence_ok": row["evidence_quote_found"],
            "source_ok": row["rag_source_hit"],
            "comment": row["judge_comment"],
        }
        for row in rows
    ]

    save_json(OUTPUT_DIR / "judge_report.json", judge_report)
    save_json(OUTPUT_DIR / "eval_summary.json", metrics)

    print("Evaluation finished.")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    failures = detailed_df[detailed_df["judge_pass"] == False]
    print("\nFailed cases:", len(failures))
    if len(failures) > 0:
        print(failures[
            [
                "case_id",
                "ticket_id",
                "expected_category",
                "predicted_category",
                "expected_priority",
                "predicted_priority",
                "expected_department",
                "predicted_department",
                "judge_comment",
            ]
        ].to_string(index=False))

    print("\nCreated files:")
    for name in [
        "eval_results.csv",
        "eval_detailed.csv",
        "eval_detailed.json",
        "judge_report.json",
        "eval_summary.json",
    ]:
        print("-", OUTPUT_DIR / name)


if __name__ == "__main__":
    main()
