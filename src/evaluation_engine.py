"""
src/evaluation_engine.py
Automated evaluation engine — 6 metrics across 3 categories.

Category A — Lexical (deterministic, free):
    - ROUGE-L  (rouge-score)
    - BLEU     (nltk sentence_bleu + smoothing)

Category B — Semantic (model-based, free on CPU):
    - BERTScore F1  (bert_score, roberta-large)

Category C — Custom LLM-as-Judge (Groq free tier):
    - Fact Recall Score    (Custom Metric 1)
    - Tone Accuracy Score  (Custom Metric 2)
    - Fluency & Prof. Score (Custom Metric 3)

Composite score uses configurable weights from config.yaml.
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass

import nltk
from bert_score import BERTScorer
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from src.models import (
    EvaluationScore,
    EvaluationWeights,
    GenerationResult,
    ReferenceEmail,
    Scenario,
)

logger = logging.getLogger(__name__)

# Download required NLTK data if not already present
for _pkg in ("punkt", "punkt_tab"):
    try:
        nltk.data.find(f"tokenizers/{_pkg}")
    except LookupError:
        nltk.download(_pkg, quiet=True)

# BERTScore model — roberta-large runs on CPU; downloaded once and cached
_BERTSCORE_MODEL = "roberta-large"

# Global instance to avoid reloading weights/checking internet per scenario
_BERT_SCORER = None

def get_bert_scorer() -> BERTScorer:
    """Lazy load BERTScorer to ensure it's loaded only when needed."""
    global _BERT_SCORER
    if _BERT_SCORER is None:
        logger.info("Initializing BERTScorer into memory. This may take a moment...")
        _BERT_SCORER = BERTScorer(model_type=_BERTSCORE_MODEL, lang="en")
    return _BERT_SCORER


# ---------------------------------------------------------------------------
# A. Lexical metrics
# ---------------------------------------------------------------------------

def compute_rouge_l(generated: str, reference: str) -> float:
    """
    Compute ROUGE-L F1 score between generated and reference email.
    Uses longest common subsequence — rewards fluent, ordered overlap.
    Range: [0.0, 1.0]
    """
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    result = scorer.score(reference, generated)
    return round(result["rougeL"].fmeasure, 4)


def compute_bleu(generated: str, reference: str) -> float:
    """
    Compute sentence-level BLEU (1–4 gram) with SmoothingFunction method1.
    Smoothing prevents zero scores when n-gram matches are sparse.
    Range: [0.0, 1.0]
    """
    reference_tokens = nltk.word_tokenize(reference.lower())
    hypothesis_tokens = nltk.word_tokenize(generated.lower())
    smoothing = SmoothingFunction().method1
    score = sentence_bleu(
        [reference_tokens],
        hypothesis_tokens,
        smoothing_function=smoothing,
    )
    return round(float(score), 4)


# ---------------------------------------------------------------------------
# B. Semantic metric
# ---------------------------------------------------------------------------

def compute_bert_score(generated: str, reference: str) -> float:
    """
    Compute BERTScore F1 using roberta-large.
    Captures semantic similarity beyond surface n-gram overlap.
    Uses globally instantiated scorer for massive performance boost.
    Range: [0.0, 1.0]
    """
    try:
        scorer = get_bert_scorer()
        _, _, f1 = scorer.score(cands=[generated], refs=[reference])
        return round(float(f1[0]), 4)
    except Exception as exc:  # noqa: BLE001
        logger.error("BERTScore failed: %s — returning 0.0", exc)
        return 0.0


# ---------------------------------------------------------------------------
# C. Composite score
# ---------------------------------------------------------------------------

def compute_composite_score(
    rouge_l: float,
    bleu: float,
    bert_score_f1: float,
    fact_recall: float,
    tone_accuracy: float,
    fluency_professionalism: float,
    weights: EvaluationWeights,
) -> float:
    """
    Weighted composite score across all 6 metrics.
    All inputs and output in [0.0, 1.0].
    """
    composite = (
        weights.rouge_weight * rouge_l
        + weights.bleu_weight * bleu
        + weights.bertscore_weight * bert_score_f1
        + weights.fact_recall_weight * fact_recall
        + weights.tone_accuracy_weight * tone_accuracy
        + weights.fluency_weight * fluency_professionalism
    )
    return round(min(max(composite, 0.0), 1.0), 4)


# ---------------------------------------------------------------------------
# Full scenario evaluation
# ---------------------------------------------------------------------------

def evaluate_scenario(
    result: GenerationResult,
    reference: ReferenceEmail,
    scenario: Scenario,
    judge_scores: dict[str, float],
    weights: EvaluationWeights,
    run_timestamp: str,
) -> EvaluationScore:
    """
    Compute all 6 metrics for a single (generated, reference, scenario) triple.

    Args:
        result:       GenerationResult from the LLM client.
        reference:    Human reference email for this scenario.
        scenario:     The original input scenario (for metadata).
        judge_scores: Pre-computed dict from llm_judge.judge_scenario().
        weights:      Metric weights from config.yaml.
        run_timestamp: ISO timestamp string for this run.

    Returns:
        Fully populated EvaluationScore.
    """
    generated = result.generated_email

    if result.status == "failed" or not generated.strip():
        logger.warning("Skipping metrics for failed generation: %s / %s", result.model_name, result.scenario_id)
        return EvaluationScore(
            scenario_id=result.scenario_id,
            model_name=result.model_name,
            rouge_l=0.0,
            bleu=0.0,
            bert_score_f1=0.0,
            fact_recall=0.0,
            tone_accuracy=0.0,
            fluency_professionalism=0.0,
            composite_score=0.0,
            run_timestamp=run_timestamp,
        )

    ref_text = reference.email_text

    # --- Lexical ---
    rouge_l = compute_rouge_l(generated, ref_text)
    bleu = compute_bleu(generated, ref_text)
    logger.info("  [metrics/%s/%s] ROUGE-L=%.3f  BLEU=%.3f", result.model_name, scenario.id, rouge_l, bleu)

    # --- Semantic ---
    logger.info("  [metrics/%s/%s] Computing BERTScore ...", result.model_name, scenario.id)
    bs_f1 = compute_bert_score(generated, ref_text)
    logger.info("  [metrics/%s/%s] BERTScore-F1=%.3f", result.model_name, scenario.id, bs_f1)

    # --- LLM-judge (passed in) ---
    fact_recall = judge_scores.get("fact_recall", 0.0)
    tone_accuracy = judge_scores.get("tone_accuracy", 0.0)
    fluency = judge_scores.get("fluency_professionalism", 0.0)

    # --- Composite ---
    composite = compute_composite_score(
        rouge_l, bleu, bs_f1, fact_recall, tone_accuracy, fluency, weights
    )

    return EvaluationScore(
        scenario_id=result.scenario_id,
        model_name=result.model_name,
        rouge_l=rouge_l,
        bleu=bleu,
        bert_score_f1=bs_f1,
        fact_recall=fact_recall,
        tone_accuracy=tone_accuracy,
        fluency_professionalism=fluency,
        composite_score=composite,
        run_timestamp=run_timestamp,
    )


def evaluate_all(
    results: list[GenerationResult],
    references: dict[str, ReferenceEmail],
    scenarios: dict[str, Scenario],
    judge_fn,
    weights: EvaluationWeights,
    run_timestamp: str,
) -> list[EvaluationScore]:
    """
    Evaluate all generation results.

    Args:
        results:       All GenerationResult objects (model A + model B).
        references:    Dict mapping scenario_id → ReferenceEmail.
        scenarios:     Dict mapping scenario_id → Scenario.
        judge_fn:      Callable(generated_email, scenario) → dict[str, float].
        weights:       Metric weights.
        run_timestamp: ISO timestamp for this run.

    Returns:
        List of EvaluationScore, one per GenerationResult.
    """
    scores: list[EvaluationScore] = []

    for i, result in enumerate(results, start=1):
        sid = result.scenario_id
        logger.info("[%d/%d] Evaluating %s / %s", i, len(results), result.model_name, sid)

        ref = references.get(sid)
        scenario = scenarios.get(sid)

        if ref is None or scenario is None:
            logger.error("Missing reference or scenario for %s — skipping.", sid)
            continue

        # LLM judge calls (Groq)
        judge_scores = judge_fn(result.generated_email, scenario)

        score = evaluate_scenario(
            result=result,
            reference=ref,
            scenario=scenario,
            judge_scores=judge_scores,
            weights=weights,
            run_timestamp=run_timestamp,
        )
        scores.append(score)

    return scores
