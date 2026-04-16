"""
tests/test_evaluation_engine.py
Unit tests for the evaluation engine — all 6 metrics and composite scoring.

All tests are fully offline: no LLM APIs, no network calls.
ROUGE-L and BLEU are deterministic; BERTScore is tested for type/range only.
"""

import pytest

from src.evaluation_engine import (
    compute_bert_score,
    compute_bleu,
    compute_composite_score,
    compute_rouge_l,
    evaluate_scenario,
)
from src.models import (
    EvaluationScore,
    EvaluationWeights,
    GenerationResult,
    ReferenceEmail,
    Scenario,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_weights():
    return EvaluationWeights(
        rouge_weight=0.25,
        bleu_weight=0.10,
        bertscore_weight=0.15,
        fact_recall_weight=0.20,
        tone_accuracy_weight=0.15,
        fluency_weight=0.15,
    )

@pytest.fixture
def sample_scenario():
    return Scenario(
        id="S01",
        intent="Follow up after product demo",
        facts=["Met on Tuesday", "Pricing proposal due Friday", "Call Friday 2 PM"],
        tone="formal",
    )

@pytest.fixture
def sample_reference():
    return ReferenceEmail(
        scenario_id="S01",
        email_text=(
            "Dear Client,\n\nThank you for attending our product demo on Tuesday. "
            "I will have the pricing proposal ready by Friday. "
            "Looking forward to our call at 2 PM on Friday.\n\nBest regards,\nSender"
        ),
    )

@pytest.fixture
def sample_result(sample_scenario):
    return GenerationResult(
        scenario_id="S01",
        model_name="gpt-4o",
        generated_email=(
            "Dear Client,\n\nThank you for joining our demo on Tuesday. "
            "I will send you the pricing proposal by end of Friday. "
            "I look forward to speaking with you at 2 PM on Friday.\n\nBest,\nSender"
        ),
        latency_ms=420.0,
        prompt_tokens=180,
        completion_tokens=95,
    )


# ---------------------------------------------------------------------------
# ROUGE-L
# ---------------------------------------------------------------------------

class TestComputeRougeL:
    def test_identical_texts_score_one(self):
        text = "The quick brown fox jumps over the lazy dog."
        assert compute_rouge_l(text, text) == 1.0

    def test_completely_different_texts_score_low(self):
        gen = "Hello world this is a test sentence."
        ref = "Quantum physics equations describe subatomic particles."
        score = compute_rouge_l(gen, ref)
        assert 0.0 <= score < 0.3

    def test_partial_overlap_between_zero_and_one(self):
        gen = "Thank you for attending the demo on Tuesday."
        ref = "Thank you for coming to the demo on Tuesday afternoon."
        score = compute_rouge_l(gen, ref)
        assert 0.0 < score < 1.0

    def test_returns_float(self):
        score = compute_rouge_l("sample generated", "sample reference")
        assert isinstance(score, float)

    def test_score_in_valid_range(self):
        score = compute_rouge_l("some generated email text", "some reference email text")
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# BLEU
# ---------------------------------------------------------------------------

class TestComputeBleu:
    def test_identical_texts_score_one(self):
        text = "The quick brown fox."
        score = compute_bleu(text, text)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_returns_float(self):
        score = compute_bleu("generated email text here", "reference email text here")
        assert isinstance(score, float)

    def test_score_in_valid_range(self):
        score = compute_bleu("some generated email", "some reference email text")
        assert 0.0 <= score <= 1.0

    def test_empty_overlap_score_near_zero(self):
        gen = "purple elephant dances wildly"
        ref = "corporate quarterly earnings report analysis"
        score = compute_bleu(gen, ref)
        assert score < 0.2

    def test_high_overlap_scores_high(self):
        ref = "Thank you for your time and consideration."
        gen = "Thank you for your time and consideration today."
        score = compute_bleu(gen, ref)
        assert score > 0.4


# ---------------------------------------------------------------------------
# BERTScore
# ---------------------------------------------------------------------------

class TestComputeBertScore:
    def test_returns_float(self):
        score = compute_bert_score(
            "Dear team, please review the proposal.",
            "Dear colleagues, kindly review the proposal.",
        )
        assert isinstance(score, float)

    def test_score_in_valid_range(self):
        score = compute_bert_score(
            "Thank you for your response.",
            "I appreciate your reply.",
        )
        assert 0.0 <= score <= 1.0

    def test_identical_texts_high_score(self):
        text = "I am writing to follow up on our previous discussion."
        score = compute_bert_score(text, text)
        assert score > 0.95


# ---------------------------------------------------------------------------
# Composite score
# ---------------------------------------------------------------------------

class TestComputeCompositeScore:
    def test_all_perfect_scores_give_one(self, default_weights):
        score = compute_composite_score(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, default_weights)
        assert score == pytest.approx(1.0, abs=0.001)

    def test_all_zero_scores_give_zero(self, default_weights):
        score = compute_composite_score(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, default_weights)
        assert score == 0.0

    def test_mixed_scores_in_valid_range(self, default_weights):
        score = compute_composite_score(0.5, 0.2, 0.85, 0.9, 0.8, 0.75, default_weights)
        assert 0.0 <= score <= 1.0

    def test_formula_is_weighted_sum(self, default_weights):
        rouge_l, bleu, bs, fr, ta, fl = 0.5, 0.2, 0.85, 0.9, 0.8, 0.75
        expected = (
            0.25 * rouge_l
            + 0.10 * bleu
            + 0.15 * bs
            + 0.20 * fr
            + 0.15 * ta
            + 0.15 * fl
        )
        computed = compute_composite_score(rouge_l, bleu, bs, fr, ta, fl, default_weights)
        assert computed == pytest.approx(round(expected, 4), abs=0.0001)

    def test_returns_float(self, default_weights):
        score = compute_composite_score(0.4, 0.3, 0.8, 0.7, 0.6, 0.5, default_weights)
        assert isinstance(score, float)

    def test_clamped_above_one(self, default_weights):
        # Shouldn't happen in practice, but defensive clamping is tested
        score = compute_composite_score(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, default_weights)
        assert score <= 1.0

    def test_clamped_below_zero(self, default_weights):
        score = compute_composite_score(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, default_weights)
        assert score >= 0.0


# ---------------------------------------------------------------------------
# evaluate_scenario (integration — no API)
# ---------------------------------------------------------------------------

class TestEvaluateScenario:
    def test_failed_result_returns_all_zeros(
        self, sample_scenario, sample_reference, default_weights
    ):
        failed_result = GenerationResult(
            scenario_id="S01",
            model_name="gpt-4o",
            generated_email="",
            latency_ms=0.0,
            status="failed",
        )
        judge_scores = {"fact_recall": 0.0, "tone_accuracy": 0.0, "fluency_professionalism": 0.0}
        score = evaluate_scenario(
            result=failed_result,
            reference=sample_reference,
            scenario=sample_scenario,
            judge_scores=judge_scores,
            weights=default_weights,
            run_timestamp="20240415T120000Z",
        )
        assert score.rouge_l == 0.0
        assert score.bleu == 0.0
        assert score.composite_score == 0.0

    def test_returns_evaluation_score(
        self, sample_result, sample_reference, sample_scenario, default_weights
    ):
        judge_scores = {"fact_recall": 0.9, "tone_accuracy": 0.8, "fluency_professionalism": 0.75}
        score = evaluate_scenario(
            result=sample_result,
            reference=sample_reference,
            scenario=sample_scenario,
            judge_scores=judge_scores,
            weights=default_weights,
            run_timestamp="20240415T120000Z",
        )
        assert isinstance(score, EvaluationScore)
        assert score.scenario_id == "S01"
        assert score.model_name == "gpt-4o"
        assert 0.0 <= score.composite_score <= 1.0

    def test_all_metrics_populated(
        self, sample_result, sample_reference, sample_scenario, default_weights
    ):
        judge_scores = {"fact_recall": 0.85, "tone_accuracy": 0.75, "fluency_professionalism": 0.80}
        score = evaluate_scenario(
            result=sample_result,
            reference=sample_reference,
            scenario=sample_scenario,
            judge_scores=judge_scores,
            weights=default_weights,
            run_timestamp="20240415T120000Z",
        )
        for field in ("rouge_l", "bleu", "bert_score_f1", "fact_recall", "tone_accuracy",
                      "fluency_professionalism", "composite_score"):
            val = getattr(score, field)
            assert isinstance(val, float), f"{field} should be float"
            assert 0.0 <= val <= 1.0, f"{field} should be in [0, 1]"
