from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from llm_client import chat_json_with_usage, get_model_name, llm_config_available
from schema import LLMJudgeResponse


PROJECT_DIR = Path(__file__).parent
OUTPUT_DIR = PROJECT_DIR / "output"
INPUT_DIR = PROJECT_DIR / "input"

EVAL_CASES_PATH = INPUT_DIR / "eval_cases.json"
CLASSIFIED_PATH = OUTPUT_DIR / "classified_tickets.json"
RAG_RESULTS_PATH = OUTPUT_DIR / "rag_results.json"
LLM_REPORT_PATH = OUTPUT_DIR / "llm_judge_report.json"
LLM_RESULTS_PATH = OUTPUT_DIR / "llm_judge_results.csv"
LLM_SUMMARY_PATH = OUTPUT_DIR / "llm_judge_summary.json"


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def key(ticket_id: int | str) -> str:
    return str(ticket_id)


def fallback_summary(reason: str) -> None:
    summary = {
        "status": "skipped",
        "reason": reason,
        "total": 0,
        "llm_judge_pass_rate": None,
    }
    save_json(LLM_REPORT_PATH, [])
    save_json(LLM_SUMMARY_PATH, summary)
    pd.DataFrame([]).to_csv(LLM_RESULTS_PATH, index=False)
    print("LLM judge skipped.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_prompt(case: dict, prediction: dict, rag_result: dict) -> str:
    return f"""
You are an evaluation judge for a support ticket routing system.

Evaluate whether the predicted routing decision is correct.
Return JSON only.

Evaluation criteria:
1. category_ok: predicted category matches the ticket meaning and expected category.
2. priority_ok: predicted priority matches expected priority.
3. department_ok: predicted department matches expected department.
4. evidence_ok: evidence quote is copied from the ticket text and supports the decision.
5. source_ok: used policy sources are relevant to the decision.
6. judge_pass: true only if the main decision is correct and evidence is grounded.

Allowed JSON schema:
{{
  "judge_pass": true,
  "category_ok": true,
  "priority_ok": true,
  "department_ok": true,
  "evidence_ok": true,
  "source_ok": true,
  "comment": "short explanation"
}}

Ticket:
{case["text"]}

Expected:
category: {case["expected_category"]}
priority: {case["expected_priority"]}
department: {case["expected_department"]}
expected_sources: {case["expected_sources"]}

Prediction:
category: {prediction["category"]}
priority: {prediction["priority"]}
department: {prediction["department"]}
summary: {prediction["summary"]}
recommended_action: {prediction["recommended_action"]}
evidence_quote: {prediction["evidence_quote"]}
used_sources: {prediction["used_sources"]}

Retrieved policy context:
{rag_result.get("policy_context", [])}
""".strip()


def main() -> None:
    if not llm_config_available():
        fallback_summary("LLM_AUTH_TOKEN is not set in .env")
        return

    eval_cases = load_json(EVAL_CASES_PATH)
    predictions = load_json(CLASSIFIED_PATH)
    rag_results = load_json(RAG_RESULTS_PATH)

    predictions_by_id = {key(item["ticket_id"]): item for item in predictions}
    rag_by_id = {key(item["ticket_id"]): item for item in rag_results}

    rows = []

    for case in eval_cases:
        ticket_key = key(case["ticket_id"])
        prediction = predictions_by_id[ticket_key]
        rag_result = rag_by_id[ticket_key]

        prompt = build_prompt(case, prediction, rag_result)
        judge_model, usage = chat_json_with_usage(prompt, response_model=LLMJudgeResponse, max_retries=3)
        judge = judge_model.model_dump()

        row = {
            "case_id": case["id"],
            "ticket_id": case["ticket_id"],
            "expected_category": case["expected_category"],
            "predicted_category": prediction["category"],
            "expected_priority": case["expected_priority"],
            "predicted_priority": prediction["priority"],
            "expected_department": case["expected_department"],
            "predicted_department": prediction["department"],
            "judge_pass": bool(judge.get("judge_pass", False)),
            "category_ok": bool(judge.get("category_ok", False)),
            "priority_ok": bool(judge.get("priority_ok", False)),
            "department_ok": bool(judge.get("department_ok", False)),
            "evidence_ok": bool(judge.get("evidence_ok", False)),
            "source_ok": bool(judge.get("source_ok", False)),
            "comment": str(judge.get("comment", "")),
            "prompt_tokens": int(usage["prompt_tokens"]),
            "completion_tokens": int(usage["completion_tokens"]),
            "total_tokens": int(usage["total_tokens"]),
            "estimated_cost_usd": float(usage["estimated_cost_usd"]),
            "llm_attempts": int(usage["llm_attempts"]),
            "cost_pricing_configured": bool(usage["cost_pricing_configured"]),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(LLM_RESULTS_PATH, index=False)

    summary = {
        "status": "completed",
        "model": get_model_name(),
        "total": int(len(df)),
        "llm_judge_pass_rate": round(float(df["judge_pass"].mean()), 4),
        "category_ok_rate": round(float(df["category_ok"].mean()), 4),
        "priority_ok_rate": round(float(df["priority_ok"].mean()), 4),
        "department_ok_rate": round(float(df["department_ok"].mean()), 4),
        "evidence_ok_rate": round(float(df["evidence_ok"].mean()), 4),
        "source_ok_rate": round(float(df["source_ok"].mean()), 4),
        "prompt_tokens": int(df["prompt_tokens"].sum()),
        "completion_tokens": int(df["completion_tokens"].sum()),
        "total_tokens": int(df["total_tokens"].sum()),
        "estimated_cost_usd": round(float(df["estimated_cost_usd"].sum()), 8),
        "avg_tokens_per_case": round(float(df["total_tokens"].mean()), 4),
        "total_llm_attempts": int(df["llm_attempts"].sum()),
        "cost_pricing_configured": bool(df["cost_pricing_configured"].any()),
    }

    save_json(LLM_REPORT_PATH, rows)
    save_json(LLM_SUMMARY_PATH, summary)
    save_json(OUTPUT_DIR / "cost_summary.json", summary)

    print("LLM judge finished.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    failed = df[df["judge_pass"] == False]
    print("\nFailed by LLM judge:", len(failed))
    if len(failed) > 0:
        print(
            failed[
                [
                    "case_id",
                    "ticket_id",
                    "expected_category",
                    "predicted_category",
                    "expected_department",
                    "predicted_department",
                    "comment",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
