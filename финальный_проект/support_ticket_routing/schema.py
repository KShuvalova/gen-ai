from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


Category = Literal[
    "account_access",
    "billing_refund",
    "technical_bug",
    "subscription",
    "security_privacy",
    "delivery_status",
    "product_question",
    "other",
]

Priority = Literal["low", "medium", "high", "urgent"]

Department = Literal[
    "support_l1",
    "support_l2",
    "billing_team",
    "technical_team",
    "security_team",
    "logistics_team",
    "retention_team",
]


class TicketInput(BaseModel):
    ticket_id: int | str
    issue_description: str
    raw_category: str | None = None
    priority: str | None = None
    status: str | None = None
    channel: str | None = None
    escalated: str | bool | None = None
    sla_breached: str | bool | None = None

    @field_validator("issue_description")
    @classmethod
    def issue_description_not_empty(cls, value: str) -> str:
        value = str(value).strip()
        if len(value) < 10:
            raise ValueError("issue_description must contain at least 10 characters")
        return value


class RetrievedPolicy(BaseModel):
    source_id: str
    score: float = Field(ge=0.0)
    text: str

    @field_validator("source_id")
    @classmethod
    def source_id_must_be_md(cls, value: str) -> str:
        value = str(value).strip()
        if not value.endswith(".md"):
            raise ValueError("source_id must point to a markdown policy document")
        return value


class TicketClassification(BaseModel):
    ticket_id: int | str
    category: Category
    priority: Priority
    department: Department
    summary: str
    recommended_action: str
    evidence_quote: str
    confidence: float = Field(ge=0.0, le=1.0)
    escalation_required: bool
    escalation_reason: str
    used_sources: list[str]
    policy_context: list[RetrievedPolicy]

    @field_validator("summary", "recommended_action", "evidence_quote")
    @classmethod
    def text_fields_not_empty(cls, value: str) -> str:
        value = str(value).strip()
        if not value:
            raise ValueError("text fields must not be empty")
        return value

    @field_validator("evidence_quote")
    @classmethod
    def evidence_quote_reasonable_length(cls, value: str) -> str:
        value = str(value).strip()
        if len(value) < 8:
            raise ValueError("evidence_quote is too short")
        if len(value) > 300:
            raise ValueError("evidence_quote is too long")
        return value

    @field_validator("used_sources")
    @classmethod
    def used_sources_not_empty(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("used_sources must not be empty")
        for item in value:
            if not item.endswith(".md"):
                raise ValueError("each source must be a markdown policy document")
        return value


class HallucinationCheck(BaseModel):
    ticket_id: int | str
    evidence_quote_found: bool
    invalid_sources: list[str]
    ghost_quote_count: int = Field(ge=0)
    fake_source_count: int = Field(ge=0)


class AgentTraceStep(BaseModel):
    step_id: int
    tool_name: str
    input_summary: str
    output_summary: str


class AgentTrace(BaseModel):
    ticket_id: int | str
    steps: list[AgentTraceStep]

    @field_validator("steps")
    @classmethod
    def steps_not_empty(cls, value: list[AgentTraceStep]) -> list[AgentTraceStep]:
        if not value:
            raise ValueError("agent trace must contain at least one step")
        return value


class JudgeResult(BaseModel):
    ticket_id: int | str
    judge_pass: bool
    category_ok: bool
    priority_ok: bool
    department_ok: bool
    evidence_ok: bool
    source_ok: bool
    comment: str


class EvalCase(BaseModel):
    id: str
    ticket_id: int | str
    text: str
    expected_category: Category
    expected_priority: Priority
    expected_department: Department
    expected_sources: list[str]
    expected_evidence_quote: str
    expected_escalation: bool


class EvalResult(BaseModel):
    total: int
    category_accuracy: float
    priority_accuracy: float
    department_accuracy: float
    escalation_accuracy: float
    quote_valid_rate: float
    rag_hit_rate: float
    ghost_quote_count: int
    fake_source_count: int
