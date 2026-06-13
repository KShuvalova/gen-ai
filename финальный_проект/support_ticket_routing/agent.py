from __future__ import annotations

from pathlib import Path
from typing import Any

from rag import PolicyRAG
from schema import AgentTrace, AgentTraceStep, TicketClassification, TicketInput
from tools import (
    choose_department,
    classify_ticket_text,
    hallucination_check,
    make_evidence_quote,
    make_summary,
    normalize_priority,
    recommended_action,
    retrieve_policy_docs,
    should_escalate,
)


def classify_and_route_ticket(
    ticket: dict[str, Any],
    rag: PolicyRAG,
    policy_docs_dir: str | Path,
) -> dict[str, Any]:
    ticket_input = TicketInput(
        ticket_id=ticket["ticket_id"],
        issue_description=ticket["issue_description"],
        raw_category=ticket.get("raw_category"),
        priority=ticket.get("priority"),
        status=ticket.get("status"),
        channel=ticket.get("channel"),
        escalated=ticket.get("escalated"),
        sla_breached=ticket.get("sla_breached"),
    )

    trace_steps: list[AgentTraceStep] = []

    category, confidence = classify_ticket_text(ticket_input.issue_description)
    trace_steps.append(
        AgentTraceStep(
            step_id=1,
            tool_name="classify_ticket_text",
            input_summary=ticket_input.issue_description[:180],
            output_summary=f"category={category}, confidence={confidence:.2f}",
        )
    )

    priority = normalize_priority(ticket.get("normalized_priority") or ticket.get("priority"))
    department = choose_department(category)
    trace_steps.append(
        AgentTraceStep(
            step_id=2,
            tool_name="choose_department",
            input_summary=f"category={category}, priority={priority}",
            output_summary=f"department={department}",
        )
    )

    escalation_required, escalation_reason = should_escalate(
        priority=priority,
        escalated=ticket.get("escalated"),
        sla_breached=ticket.get("sla_breached"),
    )
    trace_steps.append(
        AgentTraceStep(
            step_id=3,
            tool_name="should_escalate",
            input_summary=(
                f"priority={priority}, escalated={ticket.get('escalated')}, "
                f"sla_breached={ticket.get('sla_breached')}"
            ),
            output_summary=f"escalation_required={escalation_required}; reason={escalation_reason}",
        )
    )

    query = (
        f"{ticket_input.issue_description} "
        f"category {category} priority {priority} department {department} "
        f"escalation {escalation_required}"
    )
    policy_context = retrieve_policy_docs(rag, query=query, top_k=3)
    used_sources = [doc.source_id for doc in policy_context]

    trace_steps.append(
        AgentTraceStep(
            step_id=4,
            tool_name="retrieve_policy_docs",
            input_summary=query[:220],
            output_summary=f"used_sources={used_sources}",
        )
    )

    evidence_quote = make_evidence_quote(ticket_input.issue_description)
    action = recommended_action(category, priority, escalation_required)

    classification = TicketClassification(
        ticket_id=ticket_input.ticket_id,
        category=category,
        priority=priority,
        department=department,
        summary=make_summary(ticket_input.issue_description),
        recommended_action=action,
        evidence_quote=evidence_quote,
        confidence=confidence,
        escalation_required=escalation_required,
        escalation_reason=escalation_reason,
        used_sources=used_sources,
        policy_context=policy_context,
    )

    check = hallucination_check(
        ticket_id=ticket_input.ticket_id,
        ticket_text=ticket_input.issue_description,
        evidence_quote=classification.evidence_quote,
        used_sources=classification.used_sources,
        policy_docs_dir=policy_docs_dir,
    )

    trace_steps.append(
        AgentTraceStep(
            step_id=5,
            tool_name="hallucination_check",
            input_summary=f"quote={classification.evidence_quote}; sources={classification.used_sources}",
            output_summary=(
                f"evidence_quote_found={check.evidence_quote_found}, "
                f"ghost_quote_count={check.ghost_quote_count}, "
                f"fake_source_count={check.fake_source_count}"
            ),
        )
    )

    trace = AgentTrace(
        ticket_id=ticket_input.ticket_id,
        steps=trace_steps,
    )

    return {
        "classification": classification,
        "trace": trace,
        "hallucination_check": check,
    }
