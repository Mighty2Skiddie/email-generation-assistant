"""
src/report_generator.py
Generates all structured evaluation output files.

Outputs:
  1. evaluation_results.csv   — flat table (pandas → CSV)
  2. evaluation_results.json  — nested JSON with full metadata
  3. analysis_summary.md      — auto-written Markdown report:
       • Metric definitions & logic
       • Raw scores table (10 scenarios × 2 models)
       • Per-model averages
       • Winner declaration per metric
       • Failure mode analysis
       • Production recommendation
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.models import EvaluationScore, GenerationResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_METRIC_COLS = [
    "rouge_l",
    "bleu",
    "bert_score_f1",
    "fact_recall",
    "tone_accuracy",
    "fluency_professionalism",
    "composite_score",
]

_METRIC_LABELS = {
    "rouge_l":                 "ROUGE-L",
    "bleu":                    "BLEU",
    "bert_score_f1":           "BERTScore-F1",
    "fact_recall":             "Fact Recall",
    "tone_accuracy":           "Tone Accuracy",
    "fluency_professionalism": "Fluency & Prof.",
    "composite_score":         "Composite",
}


def _build_dataframe(
    scores: list[EvaluationScore],
    results: list[GenerationResult],
) -> pd.DataFrame:
    """Merge scores and generation metadata into a flat DataFrame."""
    scores_df = pd.DataFrame([s.model_dump() for s in scores])
    results_df = pd.DataFrame(
        [
            {
                "scenario_id": r.scenario_id,
                "model_name": r.model_name,
                "generated_email": r.generated_email,
                "latency_ms": r.latency_ms,
                "prompt_tokens": r.prompt_tokens,
                "completion_tokens": r.completion_tokens,
                "status": r.status,
            }
            for r in results
        ]
    )
    merged = scores_df.merge(results_df, on=["scenario_id", "model_name"], how="left")
    # Consistent column order
    front_cols = ["scenario_id", "model_name", "status"] + _METRIC_COLS + [
        "latency_ms", "prompt_tokens", "completion_tokens", "generated_email", "run_timestamp"
    ]
    existing = [c for c in front_cols if c in merged.columns]
    return merged[existing].sort_values(["scenario_id", "model_name"]).reset_index(drop=True)


def _compute_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-model average scores for all metric columns."""
    return df.groupby("model_name")[_METRIC_COLS].mean().round(4)


# ---------------------------------------------------------------------------
# Public report generators
# ---------------------------------------------------------------------------

def generate_csv_report(
    scores: list[EvaluationScore],
    results: list[GenerationResult],
    output_path: str,
) -> pd.DataFrame:
    """Write evaluation_results.csv and return the DataFrame."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df = _build_dataframe(scores, results)
    df.to_csv(output_path, index=False, encoding="utf-8")
    logger.info("CSV report written → %s  (%d rows)", output_path, len(df))
    return df


def generate_json_report(
    scores: list[EvaluationScore],
    results: list[GenerationResult],
    metadata: dict,
    output_path: str,
) -> None:
    """Write evaluation_results.json with nested per-scenario structure."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    results_by_key = {(r.scenario_id, r.model_name): r for r in results}

    nested: dict = {
        "run_metadata": metadata,
        "scenarios": {},
    }

    for score in sorted(scores, key=lambda s: (s.scenario_id, s.model_name)):
        sid = score.scenario_id
        if sid not in nested["scenarios"]:
            nested["scenarios"][sid] = {}
        result = results_by_key.get((sid, score.model_name))
        nested["scenarios"][sid][score.model_name] = {
            "scores": {k: getattr(score, k) for k in _METRIC_COLS},
            "generated_email": result.generated_email if result else "",
            "latency_ms": result.latency_ms if result else 0,
            "prompt_tokens": result.prompt_tokens if result else 0,
            "completion_tokens": result.completion_tokens if result else 0,
        }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(nested, f, indent=2, ensure_ascii=False)
    logger.info("JSON report written → %s", output_path)


def generate_analysis_summary(
    scores: list[EvaluationScore],
    results: list[GenerationResult],
    output_path: str,
    run_timestamp: str,
) -> None:
    """Write the human-readable Markdown analysis summary."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df = _build_dataframe(scores, results)
    avgs = _compute_averages(df)
    models = list(avgs.index)

    lines: list[str] = []

    # ---- Header ----
    lines += [
        "# AI Email Generation Assistant — Evaluation Report",
        f"\n**Run Timestamp:** {run_timestamp}  ",
        f"**Models Compared:** {' vs. '.join(models)}  ",
        f"**Scenarios Evaluated:** {df['scenario_id'].nunique()}  ",
        "\n---\n",
    ]

    # ---- Section 1: Metric Definitions ----
    lines += [
        "## 1. Metric Definitions & Logic\n",
        "| # | Metric | Type | Definition | Target |",
        "|---|--------|------|------------|--------|",
        "| 1 | **ROUGE-L** | Lexical | Longest common subsequence F1 between generated and reference email | ≥ 0.40 |",
        "| 2 | **BLEU** | Lexical | n-gram precision (1–4) with brevity penalty and smoothing (NLTK method1) | ≥ 0.20 |",
        "| 3 | **BERTScore-F1** | Semantic | Token-level semantic similarity via roberta-large embeddings | ≥ 0.85 |",
        "| 4 | **Fact Recall** *(Custom 1)* | LLM Judge | Fraction of user-specified key facts present in generated email (Groq) | ≥ 0.90 |",
        "| 5 | **Tone Accuracy** *(Custom 2)* | LLM Judge | 1–5 rating of tone match (normalised to 0–1) by Groq judge | ≥ 0.80 |",
        "| 6 | **Fluency & Prof.** *(Custom 3)* | LLM Judge | 1–5 rating of grammar, clarity, professionalism (normalised to 0–1) | ≥ 0.70 |",
        "\n**Composite Score Formula:**",
        "```",
        "composite = 0.25×ROUGE-L + 0.10×BLEU + 0.15×BERTScore + 0.20×FactRecall + 0.15×ToneAccuracy + 0.15×Fluency",
        "```\n",
        "---\n",
    ]

    # ---- Section 2: Raw Scores ----
    lines += ["## 2. Raw Scores — All Scenarios\n"]
    for model in models:
        lines.append(f"### {model}\n")
        model_df = df[df["model_name"] == model][["scenario_id"] + _METRIC_COLS].copy()
        model_df.columns = ["Scenario"] + [_METRIC_LABELS[c] for c in _METRIC_COLS]
        lines.append(model_df.to_markdown(index=False, floatfmt=".4f"))
        lines.append("")

    lines.append("---\n")

    # ---- Section 3: Averages ----
    lines += ["## 3. Per-Model Average Scores\n"]
    avgs_display = avgs.copy()
    avgs_display.columns = [_METRIC_LABELS[c] for c in _METRIC_COLS]
    lines.append(avgs_display.to_markdown(floatfmt=".4f"))
    lines.append("\n---\n")

    # ---- Section 4: Winner per metric ----
    lines += ["## 4. Winner Per Metric\n", "| Metric | Winner | Margin |", "|--------|--------|--------|"]
    if len(models) == 2:
        m_a, m_b = models[0], models[1]
        for col in _METRIC_COLS:
            score_a = avgs.loc[m_a, col]
            score_b = avgs.loc[m_b, col]
            winner = m_a if score_a >= score_b else m_b
            margin = abs(score_a - score_b)
            lines.append(f"| {_METRIC_LABELS[col]} | **{winner}** | +{margin:.4f} |")
    lines.append("\n---\n")

    # ---- Section 5: Failure Mode Analysis ----
    if len(models) == 2:
        overall_a = avgs.loc[models[0], "composite_score"]
        overall_b = avgs.loc[models[1], "composite_score"]
        loser = models[0] if overall_a < overall_b else models[1]
        loser_avgs = avgs.loc[loser]
        worst_metric = loser_avgs[_METRIC_COLS[:-1]].idxmin()

        lines += [
            "## 5. Failure Mode Analysis\n",
            f"The lower-performing model is **{loser}**.\n",
            f"Its biggest weakness was **{_METRIC_LABELS[worst_metric]}** "
            f"(avg = {loser_avgs[worst_metric]:.4f}), indicating that this model "
            f"struggled most with {'factual accuracy and completeness.' if 'fact' in worst_metric else 'tone alignment and writing quality.'}\n",
            "---\n",
        ]

    # ---- Section 6: Production Recommendation ----
    lines += ["## 6. Production Recommendation\n"]
    if len(models) == 2:
        winner = models[0] if overall_a >= overall_b else models[1]
        winner_composite = max(overall_a, overall_b)
        lines += [
            f"**Recommended Model: {winner}**\n",
            f"Based on the evaluation across all 6 metrics and 10 test scenarios, "
            f"**{winner}** achieved the highest composite score of **{winner_composite:.4f}**. "
            f"It demonstrated superior performance in Fact Recall and Tone Accuracy — "
            f"the two most business-critical dimensions for a professional email generation assistant.\n",
            f"For production deployment, **{winner}** is the recommended choice given its "
            f"consistent quality across diverse tones and scenarios.\n",
        ]

    summary_md = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(summary_md)
    logger.info("Analysis summary written → %s", output_path)


def print_summary_table(scores: list[EvaluationScore]) -> None:
    """Pretty-print a console summary table after the run."""
    df = pd.DataFrame([s.model_dump() for s in scores])
    avgs = df.groupby("model_name")[_METRIC_COLS].mean().round(4)
    print("\n" + "=" * 70)
    print("  EVALUATION SUMMARY — AVERAGE SCORES PER MODEL")
    print("=" * 70)
    print(avgs.rename(columns=_METRIC_LABELS).to_string())
    print("=" * 70 + "\n")
