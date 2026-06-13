from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from agent import classify_and_route_ticket
from rag import PolicyRAG


PROJECT_DIR = Path(__file__).parent
POLICY_DOCS_DIR = PROJECT_DIR / "input" / "policy_docs"
OUTPUT_DIR = PROJECT_DIR / "output"
INPUT_PATH = PROJECT_DIR / "input" / "challenge_cases.json"


CHALLENGE_CASES = [
    {
        "id": "challenge_001",
        "ticket_id": "challenge_001",
        "issue_description": "My account was locked after too many attempts and the password reset email never arrives.",
        "priority": "High",
        "expected_category": "account_access",
        "expected_department": "support_l2",
    },
    {
        "id": "challenge_002",
        "ticket_id": "challenge_002",
        "issue_description": "I was charged twice after the checkout page crashed during payment.",
        "priority": "High",
        "expected_category": "billing_refund",
        "expected_department": "billing_team",
    },
    {
        "id": "challenge_003",
        "ticket_id": "challenge_003",
        "issue_description": "Since the latest update the app is slow and reports do not load.",
        "priority": "Medium",
        "expected_category": "technical_bug",
        "expected_department": "technical_team",
    },
    {
        "id": "challenge_004",
        "ticket_id": "challenge_004",
        "issue_description": "Please cancel my trial renewal because I cannot find the plan settings.",
        "priority": "Medium",
        "expected_category": "subscription",
        "expected_department": "support_l2",
    },
    {
        "id": "challenge_005",
        "ticket_id": "challenge_005",
        "issue_description": "Someone used my account without permission and changed my email address.",
        "priority": "Urgent",
        "expected_category": "security_privacy",
        "expected_department": "security_team",
    },
    {
        "id": "challenge_006",
        "ticket_id": "challenge_006",
        "issue_description": "Where can I find the export feature for invoices?",
        "priority": "Low",
        "expected_category": "product_question",
        "expected_department": "support_l1",
    },
    {
        "id": "challenge_007",
        "ticket_id": "challenge_007",
        "issue_description": "The package tracking number is not updating and the shipment is delayed.",
        "priority": "Medium",
        "expected_category": "delivery_status",
        "expected_department": "logistics_team",
    },
    {
        "id": "challenge_008",
        "ticket_id": "challenge_008",
        "issue_description": "My subscription was cancelled and the latest invoice still charged my card.",
        "priority": "High",
        "expected_category": "subscription",
        "expected_department": "support_l2",
    },
    {
        "id": "challenge_009",
        "ticket_id": "challenge_009",
        "issue_description": "I want all my personal data deleted from your platform.",
        "priority": "High",
        "expected_category": "security_privacy",
        "expected_department": "security_team",
    },
    {
        "id": "challenge_010",
        "ticket_id": "challenge_010",
        "issue_description": "The upload button returns an error, but only when I use the free plan.",
        "priority": "Medium",
        "expected_category": "technical_bug",
        "expected_department": "technical_team",
    },
    {
        "id": "challenge_011",
        "ticket_id": "challenge_011",
        "issue_description": "My bank account was charged, but I cannot access the account dashboard to see the receipt.",
        "priority": "High",
        "expected_category": "billing_refund",
        "expected_department": "billing_team",
    },
    {
        "id": "challenge_012",
        "ticket_id": "challenge_012",
        "issue_description": "The system is not syncing data, and my subscription status also looks wrong.",
        "priority": "Medium",
        "expected_category": "technical_bug",
        "expected_department": "technical_team",
    },
]


def save_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    save_json(INPUT_PATH, CHALLENGE_CASES)

    rag = PolicyRAG(POLICY_DOCS_DIR)

    rows = []
    predictions = []

    for case in CHALLENGE_CASES:
        ticket = {
            "ticket_id": case["ticket_id"],
            "issue_description": case["issue_description"],
            "priority": case["priority"],
            "normalized_priority": case["priority"].lower(),
            "raw_category": "challenge_case",
            "status": "Open",
            "channel": "Web Form",
            "escalated": "No",
            "sla_breached": "No",
        }

        result = classify_and_route_ticket(
            ticket=ticket,
            rag=rag,
            policy_docs_dir=POLICY_DOCS_DIR,
        )

        classification = result["classification"].model_dump()
        hallucination = result["hallucination_check"].model_dump()
        trace = result["trace"].model_dump()

        category_ok = classification["category"] == case["expected_category"]
        department_ok = classification["department"] == case["expected_department"]
        quote_ok = hallucination["evidence_quote_found"]
        sources_ok = hallucination["fake_source_count"] == 0
        path_ok = len(trace["steps"]) == 5

        passed = category_ok and department_ok and quote_ok and sources_ok and path_ok

        rows.append(
            {
                "case_id": case["id"],
                "ticket_id": case["ticket_id"],
                "issue_description": case["issue_description"],
                "expected_category": case["expected_category"],
                "predicted_category": classification["category"],
                "category_ok": category_ok,
                "expected_department": case["expected_department"],
                "predicted_department": classification["department"],
                "department_ok": department_ok,
                "predicted_priority": classification["priority"],
                "used_sources": classification["used_sources"],
                "quote_ok": quote_ok,
                "fake_source_count": hallucination["fake_source_count"],
                "ghost_quote_count": hallucination["ghost_quote_count"],
                "tool_steps": len(trace["steps"]),
                "passed": passed,
            }
        )

        predictions.append(
            {
                "case": case,
                "classification": classification,
                "hallucination_check": hallucination,
                "trace": trace,
            }
        )

    df = pd.DataFrame(rows)

    metrics = {
        "total": int(len(df)),
        "category_accuracy": round(float(df["category_ok"].mean()), 4),
        "department_accuracy": round(float(df["department_ok"].mean()), 4),
        "quote_valid_rate": round(float(df["quote_ok"].mean()), 4),
        "fake_source_count": int(df["fake_source_count"].sum()),
        "ghost_quote_count": int(df["ghost_quote_count"].sum()),
        "pass_rate": round(float(df["passed"].mean()), 4),
        "failed_cases": int((~df["passed"]).sum()),
    }

    df.to_csv(OUTPUT_DIR / "challenge_eval.csv", index=False)
    save_json(OUTPUT_DIR / "challenge_eval.json", predictions)
    save_json(OUTPUT_DIR / "challenge_summary.json", metrics)

    print("Challenge evaluation finished.")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    failures = df[df["passed"] == False]
    print("\nFailed challenge cases:", len(failures))
    if len(failures) > 0:
        print(
            failures[
                [
                    "case_id",
                    "expected_category",
                    "predicted_category",
                    "expected_department",
                    "predicted_department",
                    "used_sources",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
