"""
tests/test_models.py
Unit tests for Pydantic data models — input validation and schema enforcement.
"""

import pytest
from pydantic import ValidationError

from src.models import (
    EvaluationScore,
    EvaluationWeights,
    GenerationResult,
    ModelConfig,
    ReferenceEmail,
    Scenario,
)


# ---------------------------------------------------------------------------
# Scenario validation
# ---------------------------------------------------------------------------

class TestScenario:
    def test_valid_scenario_formal(self):
        s = Scenario(
            id="S01",
            intent="Follow up after meeting",
            facts=["Met on Monday", "Sent proposal"],
            tone="formal",
        )
        assert s.id == "S01"
        assert s.tone == "formal"
        assert len(s.facts) == 2

    def test_valid_scenario_all_tones(self):
        for tone in ("formal", "casual", "urgent", "empathetic"):
            s = Scenario(id="TX", intent="Test intent", facts=["fact1"], tone=tone)
            assert s.tone == tone

    def test_invalid_tone_raises(self):
        with pytest.raises(ValidationError):
            Scenario(id="S01", intent="Test", facts=["f1"], tone="aggressive")

    def test_empty_intent_raises(self):
        with pytest.raises(ValidationError):
            Scenario(id="S01", intent="   ", facts=["f1"], tone="formal")

    def test_empty_facts_list_raises(self):
        with pytest.raises(ValidationError):
            Scenario(id="S01", intent="Some intent", facts=[], tone="formal")

    def test_facts_with_blank_strings_cleaned(self):
        s = Scenario(id="S01", intent="Valid intent test", facts=["  ", "real fact", "  "], tone="casual")
        assert s.facts == ["real fact"]

    def test_all_blank_facts_raises(self):
        with pytest.raises(ValidationError):
            Scenario(id="S01", intent="Valid intent test", facts=["  ", "  "], tone="casual")

    def test_intent_stripped(self):
        s = Scenario(id="S01", intent="  Trimmed intent  ", facts=["fact"], tone="urgent")
        assert s.intent == "Trimmed intent"

    def test_missing_required_field_raises(self):
        with pytest.raises(ValidationError):
            Scenario(id="S01", facts=["fact"], tone="formal")  # missing intent


# ---------------------------------------------------------------------------
# GenerationResult validation
# ---------------------------------------------------------------------------

class TestGenerationResult:
    def test_valid_success(self):
        r = GenerationResult(
            scenario_id="S01",
            model_name="gpt-4o",
            generated_email="Dear Sir, ...",
            latency_ms=350.5,
            prompt_tokens=120,
            completion_tokens=200,
        )
        assert r.status == "success"
        assert r.error_message is None

    def test_failed_status(self):
        r = GenerationResult(
            scenario_id="S01",
            model_name="gpt-4o",
            generated_email="",
            latency_ms=100.0,
            status="failed",
            error_message="Timeout",
        )
        assert r.status == "failed"
        assert r.error_message == "Timeout"

    def test_negative_latency_raises(self):
        with pytest.raises(ValidationError):
            GenerationResult(
                scenario_id="S01",
                model_name="gpt-4o",
                generated_email="Email text",
                latency_ms=-1.0,
            )


# ---------------------------------------------------------------------------
# EvaluationScore validation
# ---------------------------------------------------------------------------

class TestEvaluationScore:
    def _valid_score(self, **overrides):
        defaults = dict(
            scenario_id="S01",
            model_name="gpt-4o",
            rouge_l=0.45,
            bleu=0.22,
            bert_score_f1=0.88,
            fact_recall=0.91,
            tone_accuracy=0.82,
            fluency_professionalism=0.76,
            composite_score=0.71,
        )
        defaults.update(overrides)
        return EvaluationScore(**defaults)

    def test_valid_score(self):
        s = self._valid_score()
        assert 0.0 <= s.composite_score <= 1.0

    def test_score_below_zero_raises(self):
        with pytest.raises(ValidationError):
            self._valid_score(rouge_l=-0.01)

    def test_score_above_one_raises(self):
        with pytest.raises(ValidationError):
            self._valid_score(bleu=1.001)


# ---------------------------------------------------------------------------
# EvaluationWeights validation
# ---------------------------------------------------------------------------

class TestEvaluationWeights:
    def test_valid_weights(self):
        w = EvaluationWeights(
            rouge_weight=0.25,
            bleu_weight=0.10,
            bertscore_weight=0.15,
            fact_recall_weight=0.20,
            tone_accuracy_weight=0.15,
            fluency_weight=0.15,
        )
        assert w.rouge_weight == 0.25

    def test_negative_weight_raises(self):
        with pytest.raises(ValidationError):
            EvaluationWeights(
                rouge_weight=-0.1,
                bleu_weight=0.10,
                bertscore_weight=0.15,
                fact_recall_weight=0.20,
                tone_accuracy_weight=0.15,
                fluency_weight=0.15,
            )


# ---------------------------------------------------------------------------
# ModelConfig
# ---------------------------------------------------------------------------

class TestModelConfig:
    def test_valid_model_config(self):
        mc = ModelConfig(provider="openai", name="gpt-4o", temperature=0.3, max_tokens=600)
        assert mc.name == "gpt-4o"

    def test_invalid_temperature_raises(self):
        with pytest.raises(ValidationError):
            ModelConfig(provider="openai", name="gpt-4o", temperature=3.0, max_tokens=600)

    def test_zero_max_tokens_raises(self):
        with pytest.raises(ValidationError):
            ModelConfig(provider="openai", name="gpt-4o", temperature=0.3, max_tokens=0)
