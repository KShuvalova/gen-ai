from __future__ import annotations

import re
from pathlib import Path

from rag import PolicyRAG
from schema import HallucinationCheck, RetrievedPolicy


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


CATEGORY_KEYWORDS = {
    "security_privacy": [
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
    "billing_refund": [
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
    "subscription": [
        "subscription",
        "cancelled",
        "canceled",
        "renewal",
        "plan",
        "trial",
        "downgrade",
        "upgrade",
    ],
    "account_access": [
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
    "technical_bug": [
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
    "delivery_status": [
        "delivery",
        "shipment",
        "shipping",
        "tracking",
        "package",
        "courier",
        "delayed order",
    ],
    "product_question": [
        "feature request",
        "feature",
        "how do i",
        "how can i",
        "question",
        "would like to know",
    ],
}


def normalize_priority(value: str | None) -> str:
    value = str(value or "").strip().lower()
    if value in {"low", "medium", "high", "urgent"}:
        return value
    return "medium"


def classify_ticket_text(text: str) -> tuple[str, float]:
    text_norm = str(text).lower()

    best_category = "other"
    best_hits = 0

    for category, keywords in CATEGORY_KEYWORDS.items():
        hits = sum(1 for keyword in keywords if keyword in text_norm)
        if hits > best_hits:
            best_category = category
            best_hits = hits

    if best_hits == 0:
        return "other", 0.55

    confidence = min(0.95, 0.65 + best_hits * 0.1)
    return best_category, confidence


def choose_department(category: str) -> str:
    return CATEGORY_TO_DEPARTMENT.get(category, "support_l1")


def make_summary(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text[:240]


def make_evidence_quote(text: str) -> str:
    text = str(text).strip()
    parts = re.split(r"(?<=[.!?])\s+", text)
    quote = parts[0].strip() if parts else text[:200].strip()
    return quote[:220]


def should_escalate(priority: str, escalated: str | bool | None, sla_breached: str | bool | None) -> tuple[bool, str]:
    priority = normalize_priority(priority)

    escalated_text = str(escalated).strip().lower()
    sla_text = str(sla_breached).strip().lower()

    reasons = []

    if priority in {"high", "urgent"}:
        reasons.append(f"priority is {priority}")

    if escalated_text in {"yes", "true", "1"}:
        reasons.append("ticket is already marked as escalated")

    if sla_text in {"yes", "true", "1"}:
        reasons.append("SLA is breached")

    if reasons:
        return True, "; ".join(reasons)

    return False, "no escalation trigger found"


def recommended_action(category: str, priority: str, escalation_required: bool) -> str:
    prefix = "Escalate and " if escalation_required else ""

    actions = {
        "account_access": "verify authentication logs, check recovery channel delivery, and guide the customer through secure account recovery.",
        "billing_refund": "check transaction status, verify billing records, and start a refund or payment investigation if needed.",
        "technical_bug": "collect reproduction details, identify the affected feature, and route the issue to the technical team.",
        "subscription": "check subscription status, verify cancellation or renewal events, and clarify the customer's plan state.",
        "security_privacy": "review security indicators, protect the account, and route the request to the security team.",
        "delivery_status": "check shipment or tracking status and route the issue to the logistics team.",
        "product_question": "answer the product question or route it to first-line support.",
        "other": "review the ticket manually and request additional information if the issue is unclear.",
    }

    return prefix + actions.get(category, actions["other"])


def retrieve_policy_docs(rag: PolicyRAG, query: str, top_k: int = 3) -> list[RetrievedPolicy]:
    return rag.search(query=query, top_k=top_k)


def check_quote_grounding(ticket_text: str, evidence_quote: str) -> bool:
    ticket_norm = re.sub(r"\s+", " ", str(ticket_text).strip().lower())
    quote_norm = re.sub(r"\s+", " ", str(evidence_quote).strip().lower())
    return bool(quote_norm) and quote_norm in ticket_norm


def check_sources_exist(used_sources: list[str], policy_docs_dir: str | Path) -> list[str]:
    docs_dir = Path(policy_docs_dir)
    existing = {path.name for path in docs_dir.glob("*.md")}
    return [source for source in used_sources if source not in existing]


def hallucination_check(
    ticket_id: int | str,
    ticket_text: str,
    evidence_quote: str,
    used_sources: list[str],
    policy_docs_dir: str | Path,
) -> HallucinationCheck:
    quote_found = check_quote_grounding(ticket_text, evidence_quote)
    invalid_sources = check_sources_exist(used_sources, policy_docs_dir)

    return HallucinationCheck(
        ticket_id=ticket_id,
        evidence_quote_found=quote_found,
        invalid_sources=invalid_sources,
        ghost_quote_count=0 if quote_found else 1,
        fake_source_count=len(invalid_sources),
    )
