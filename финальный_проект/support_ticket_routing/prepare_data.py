from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).parent
RAW_PATH = PROJECT_DIR / "input" / "raw" / "customer_support_tickets_200k.csv"
TICKETS_PATH = PROJECT_DIR / "input" / "tickets.csv"
EVAL_PATH = PROJECT_DIR / "input" / "eval_cases.json"
PROFILE_PATH = PROJECT_DIR / "input" / "dataset_profile.json"

RANDOM_STATE = 42
N_TICKETS = 100
N_EVAL = 20


CATEGORY_TO_DEPARTMENT = {
    "account_access": "support_l2",
    "billing_refund": "billing_team",
    "technical_bug": "technical_team",
    "subscription": "support_l2",
    "security_privacy": "security_team",
    "delivery_status": "logistics_team",
    "product_question": "support_l1",
    "other": "support_l1",
}


CATEGORY_TO_SOURCE = {
    "account_access": "account_access.md",
    "billing_refund": "billing_refunds.md",
    "technical_bug": "technical_bugs.md",
    "subscription": "subscription_policy.md",
    "security_privacy": "security_privacy.md",
    "delivery_status": "delivery_status.md",
    "product_question": "product_questions.md",
    "other": "routing_rules.md",
}


def normalize_priority(value: str) -> str:
    value = str(value).strip().lower()

    mapping = {
        "low": "low",
        "medium": "medium",
        "high": "high",
        "urgent": "urgent",
    }

    return mapping.get(value, "medium")


def classify_by_text(text: str) -> str:
    text_norm = str(text).lower()

    rules = [
        (
            "security_privacy",
            [
                "unauthorized",
                "hacked",
                "security",
                "privacy",
                "breach",
                "fraud",
                "suspicious",
                "personal data",
                "delete my data",
            ],
        ),
        (
            "billing_refund",
            [
                "payment",
                "billing",
                "bank account",
                "transaction",
                "charge",
                "charged",
                "refund",
                "invoice",
                "credit card",
                "statement",
                "duplicate charge",
            ],
        ),
        (
            "subscription",
            [
                "subscription",
                "cancelled",
                "canceled",
                "renewal",
                "plan",
                "trial",
                "downgrade",
                "upgrade",
            ],
        ),
        (
            "account_access",
            [
                "login",
                "log in",
                "access my account",
                "unable to access",
                "password",
                "credentials",
                "two-factor",
                "2fa",
                "authentication",
                "verification code",
            ],
        ),
        (
            "technical_bug",
            [
                "bug",
                "crash",
                "crashes",
                "error",
                "upload",
                "sync",
                "syncing",
                "report generation",
                "latest update",
                "not working",
                "performance",
                "slow",
                "timeout",
            ],
        ),
        (
            "delivery_status",
            [
                "delivery",
                "shipment",
                "shipping",
                "tracking",
                "package",
                "courier",
                "delayed order",
            ],
        ),
        (
            "product_question",
            [
                "feature request",
                "feature",
                "how do i",
                "how can i",
                "question",
                "would like to know",
            ],
        ),
    ]

    for category, keywords in rules:
        if any(keyword in text_norm for keyword in keywords):
            return category

    return "other"


def build_expected_sources(category: str, priority: str, escalated: str, sla_breached: str) -> list[str]:
    sources = ["routing_rules.md", "priority_rules.md", CATEGORY_TO_SOURCE.get(category, "routing_rules.md")]

    if priority in {"high", "urgent"} or str(escalated).lower() == "yes" or str(sla_breached).lower() == "yes":
        sources.append("escalation_policy.md")

    return list(dict.fromkeys(sources))


def make_quote(text: str) -> str:
    text = str(text).strip()
    parts = re.split(r"(?<=[.!?])\s+", text)
    quote = parts[0].strip() if parts else text[:160].strip()
    return quote[:220]


def sample_balanced(df: pd.DataFrame, n: int) -> pd.DataFrame:
    categories = sorted(df["normalized_category"].unique())
    per_category = max(1, n // max(1, len(categories)))

    parts = []

    for category in categories:
        group = df[df["normalized_category"] == category]
        if len(group) == 0:
            continue
        take = min(per_category, len(group))
        parts.append(group.sample(n=take, random_state=RANDOM_STATE))

    sampled = pd.concat(parts, ignore_index=True)

    if len(sampled) < n:
        remaining = df.drop(index=sampled.index, errors="ignore")
        extra = remaining.sample(n=min(n - len(sampled), len(remaining)), random_state=RANDOM_STATE)
        sampled = pd.concat([sampled, extra], ignore_index=True)

    if len(sampled) > n:
        sampled = sampled.sample(n=n, random_state=RANDOM_STATE).reset_index(drop=True)

    return sampled.reset_index(drop=True)


def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw dataset not found: {RAW_PATH}")

    df = pd.read_csv(RAW_PATH)

    df = df.dropna(subset=["issue_description"]).copy()
    df["issue_description"] = df["issue_description"].astype(str).str.strip()
    df = df[df["issue_description"].str.len() > 20].copy()

    df["raw_category"] = df["category"].astype(str)
    df["normalized_category"] = df["issue_description"].apply(classify_by_text)
    df["normalized_priority"] = df["priority"].apply(normalize_priority)
    df["expected_department"] = df["normalized_category"].map(CATEGORY_TO_DEPARTMENT).fillna("support_l1")

    df["expected_sources"] = df.apply(
        lambda row: build_expected_sources(
            row["normalized_category"],
            row["normalized_priority"],
            row.get("escalated", "No"),
            row.get("sla_breached", "No"),
        ),
        axis=1,
    )

    safe_columns = [
        "ticket_id",
        "product",
        "raw_category",
        "normalized_category",
        "issue_description",
        "resolution_notes",
        "priority",
        "normalized_priority",
        "status",
        "channel",
        "region",
        "subscription_type",
        "previous_tickets",
        "customer_satisfaction_score",
        "first_response_time_hours",
        "resolution_time_hours",
        "escalated",
        "sla_breached",
        "issue_complexity_score",
        "customer_segment",
        "expected_department",
        "expected_sources",
    ]

    tickets = sample_balanced(df[safe_columns], N_TICKETS)
    tickets.to_csv(TICKETS_PATH, index=False)

    eval_df = sample_balanced(tickets, N_EVAL)

    eval_cases = []
    for idx, row in eval_df.iterrows():
        eval_cases.append(
            {
                "id": f"case_{idx + 1:03d}",
                "ticket_id": int(row["ticket_id"]),
                "text": row["issue_description"],
                "expected_category": row["normalized_category"],
                "expected_priority": row["normalized_priority"],
                "expected_department": row["expected_department"],
                "expected_sources": row["expected_sources"],
                "expected_evidence_quote": make_quote(row["issue_description"]),
                "expected_escalation": bool(
                    row["normalized_priority"] in {"high", "urgent"}
                    or str(row["escalated"]).lower() == "yes"
                    or str(row["sla_breached"]).lower() == "yes"
                ),
            }
        )

    with EVAL_PATH.open("w", encoding="utf-8") as file:
        json.dump(eval_cases, file, ensure_ascii=False, indent=2)

    profile = {
        "raw_rows": int(len(df)),
        "tickets_sample_rows": int(len(tickets)),
        "eval_cases": int(len(eval_cases)),
        "ticket_categories": tickets["normalized_category"].value_counts().to_dict(),
        "ticket_priorities": tickets["normalized_priority"].value_counts().to_dict(),
        "departments": tickets["expected_department"].value_counts().to_dict(),
        "pii_removed": ["customer_name", "customer_email"],
        "source_dataset_file": RAW_PATH.name,
    }

    with PROFILE_PATH.open("w", encoding="utf-8") as file:
        json.dump(profile, file, ensure_ascii=False, indent=2)

    print("Created:")
    print("-", TICKETS_PATH.relative_to(PROJECT_DIR))
    print("-", EVAL_PATH.relative_to(PROJECT_DIR))
    print("-", PROFILE_PATH.relative_to(PROJECT_DIR))
    print("\nProfile:")
    print(json.dumps(profile, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
