from __future__ import annotations

from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


AspectName = Literal[
    "performance",
    "design",
    "support",
    "price",
    "ads",
    "reliability",
]

Sentiment = Literal[
    "positive",
    "neutral",
    "negative",
    "mixed",
    "not_mentioned",
]

IssueCategory = Literal[
    "performance",
    "design",
    "support",
    "price",
    "ads",
    "reliability",
    "other",
]

SupportLabel = Literal[
    "supported",
    "weakly_supported",
    "not_supported",
]


class Issue(BaseModel):
    category: IssueCategory
    description: str = Field(min_length=5, max_length=400)
    evidence_quote: str = Field(min_length=3, max_length=500)


class Review(BaseModel):
    review_id: str = Field(min_length=2, max_length=30)
    source_file: str
    app_name: str = Field(min_length=2, max_length=80)
    source: Literal["App Store", "Google Play", "RuStore", "Other"]
    rating: int = Field(ge=1, le=5)
    review_date: Optional[date] = None
    review_text: str = Field(min_length=10, max_length=3000)
    issues: list[Issue] = Field(default_factory=list)

    @field_validator("review_date")
    @classmethod
    def review_date_not_future(cls, value: Optional[date]) -> Optional[date]:
        if value is not None and value > date.today():
            raise ValueError("Дата отзыва не может быть позже сегодняшнего дня")
        return value


class ReviewBatch(BaseModel):
    reviews: list[Review]


class AspectScore(BaseModel):
    review_id: str
    aspect: AspectName
    sentiment: Sentiment
    severity: int = Field(ge=0, le=5)
    evidence_quote: Optional[str] = Field(default=None, max_length=500)


class AspectBatch(BaseModel):
    aspects: list[AspectScore]


class MapSummary(BaseModel):
    source_file: str
    key_findings: list[str] = Field(min_length=3, max_length=10)
    frequent_issues: list[str] = Field(min_length=2, max_length=8)
    representative_quotes: list[str] = Field(min_length=2, max_length=8)


class FinalSummary(BaseModel):
    executive_summary: str = Field(min_length=100, max_length=2500)
    main_issues: list[str] = Field(min_length=3, max_length=10)
    aspect_takeaways: list[str] = Field(min_length=3, max_length=10)
    action_items: list[str] = Field(min_length=3, max_length=10)
    evidence_quotes: list[str] = Field(min_length=3, max_length=12)


class JudgeItem(BaseModel):
    action_item: str
    support: SupportLabel
    reason: str


class JudgeReport(BaseModel):
    overall_score: float = Field(ge=0, le=1)
    factuality_score: float = Field(ge=0, le=1)
    usefulness_score: float = Field(ge=0, le=1)
    quote_grounding_score: float = Field(ge=0, le=1)
    verdict: Literal["pass", "revise"]
    weak_or_unsupported_items: list[JudgeItem]
    recommendations: list[str]
