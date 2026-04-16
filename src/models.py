"""
src/models.py
Pydantic v2 schemas for strict input validation and structured data transfer
across the entire evaluation pipeline.
"""

from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Input Schemas
# ---------------------------------------------------------------------------

ALLOWED_TONES = Literal["formal", "casual", "urgent", "empathetic"]


class Scenario(BaseModel):
    """A single test scenario — the ground-truth input to the pipeline."""

    id: str = Field(..., description="Unique scenario identifier, e.g. 'S01'")
    intent: str = Field(..., min_length=5, description="Core purpose of the email")
    facts: list[str] = Field(..., min_length=1, description="Key facts to include")
    tone: ALLOWED_TONES = Field(..., description="Desired tone of the email")

    @field_validator("intent")
    @classmethod
    def intent_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("intent must not be blank")
        return v.strip()

    @field_validator("facts")
    @classmethod
    def facts_not_empty(cls, v: list[str]) -> list[str]:
        cleaned = [f.strip() for f in v if f.strip()]
        if not cleaned:
            raise ValueError("facts must contain at least one non-empty item")
        return cleaned


class ReferenceEmail(BaseModel):
    """Human-written reference email for a scenario (ground truth for scoring)."""

    scenario_id: str
    email_text: str = Field(..., min_length=10)


# ---------------------------------------------------------------------------
# Output Schemas
# ---------------------------------------------------------------------------

class GenerationResult(BaseModel):
    """Captures the output of a single LLM email generation call."""

    scenario_id: str
    model_name: str
    generated_email: str
    latency_ms: float = Field(..., ge=0)
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    status: Literal["success", "failed"] = "success"
    error_message: Optional[str] = None


class EvaluationScore(BaseModel):
    """All metric scores for one generated email."""

    scenario_id: str
    model_name: str
    # --- Lexical metrics ---
    rouge_l: float = Field(..., ge=0.0, le=1.0)
    bleu: float = Field(..., ge=0.0, le=1.0)
    # --- Semantic metric ---
    bert_score_f1: float = Field(..., ge=0.0, le=1.0)
    # --- Custom LLM-judge metrics ---
    fact_recall: float = Field(..., ge=0.0, le=1.0)
    tone_accuracy: float = Field(..., ge=0.0, le=1.0)
    fluency_professionalism: float = Field(..., ge=0.0, le=1.0)
    # --- Composite ---
    composite_score: float = Field(..., ge=0.0, le=1.0)
    run_timestamp: str = ""


# ---------------------------------------------------------------------------
# Config Schema
# ---------------------------------------------------------------------------

class ModelConfig(BaseModel):
    """LLM model configuration loaded from config.yaml."""

    provider: str
    name: str
    temperature: float = Field(..., ge=0.0, le=2.0)
    max_tokens: int = Field(..., gt=0)


class EvaluationWeights(BaseModel):
    """Weights for the composite score formula."""

    rouge_weight: float
    bleu_weight: float
    bertscore_weight: float
    fact_recall_weight: float
    tone_accuracy_weight: float
    fluency_weight: float

    @field_validator("rouge_weight", "bleu_weight", "bertscore_weight",
                     "fact_recall_weight", "tone_accuracy_weight", "fluency_weight")
    @classmethod
    def weights_positive(cls, v: float) -> float:
        if v < 0:
            raise ValueError("All weights must be non-negative")
        return v
