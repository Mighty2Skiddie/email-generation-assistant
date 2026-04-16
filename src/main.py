"""
src/main.py
Full pipeline orchestrator — CLI entry point.

Usage:
    python -m src.main                   # Full pipeline (generate + evaluate + report)
    python -m src.main --model-a-only    # Only run Model A generation
    python -m src.main --skip-generation # Re-evaluate using results already in DB
    python -m src.main --report-only     # Regenerate reports from latest DB run

Pipeline Steps:
    1.  Load config.yaml + .env
    2.  Initialise SQLite database
    3.  Load & validate 10 scenarios (Pydantic)
    4.  Load reference emails
    5.  Build LangChain ChatPromptTemplate (few-shot + role-play)
    6.  Create LCEL generation chains (prompt | ChatOpenAI | StrOutputParser)
    7.  Create LCEL judge chains (prompt | ChatGroq | StrOutputParser)
    8.  Generate emails — Model A (gpt-4o) × 10 scenarios
    9.  Generate emails — Model B (gpt-4o-mini) × 10 scenarios
    10. Run evaluation engine (ROUGE-L, BLEU, BERTScore, 3 LLM-judge metrics)
    11. Store all results in SQLite
    12. Generate CSV + JSON + Markdown reports
    13. Print summary table to console
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Bootstrap: load env variables before any LangChain imports
# ---------------------------------------------------------------------------
load_dotenv()

from src import db  # noqa: E402
from src.evaluation_engine import evaluate_all  # noqa: E402
from src.generation_client import generate_batch  # noqa: E402
from src.llm_judge import (  # noqa: E402
    build_fact_recall_chain,
    build_fluency_chain,
    build_tone_accuracy_chain,
    create_judge_llm,
    judge_scenario,
)
from src.models import (  # noqa: E402
    EvaluationWeights,
    ModelConfig,
    ReferenceEmail,
    Scenario,
)
from src.report_generator import (  # noqa: E402
    generate_analysis_summary,
    generate_csv_report,
    generate_json_report,
    print_summary_table,
)

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
import os

Path("outputs").mkdir(exist_ok=True)

# Suppress HuggingFace symlinks and token warnings at the start
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("outputs/pipeline.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# Silence noisy third-party loggers (OpenAI/Groq/HuggingFace API calls)
logging.getLogger("httpx").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_scenarios(path: str = "data/scenarios.json") -> dict[str, Scenario]:
    """Load and Pydantic-validate all 10 test scenarios."""
    with open(path, "r", encoding="utf-8") as f:
        raw: list[dict] = json.load(f)

    scenarios: dict[str, Scenario] = {}
    for item in raw:
        try:
            s = Scenario(**item)
            scenarios[s.id] = s
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping invalid scenario %s: %s", item.get("id", "?"), exc)

    logger.info("Loaded %d valid scenarios.", len(scenarios))
    return scenarios


def load_references(path: str = "data/reference_emails.json") -> dict[str, ReferenceEmail]:
    """Load all 10 human reference emails."""
    with open(path, "r", encoding="utf-8") as f:
        raw: dict = json.load(f)

    references: dict[str, ReferenceEmail] = {}
    for sid, data in raw.items():
        references[sid] = ReferenceEmail(
            scenario_id=sid,
            email_text=data["email_text"],
        )
    logger.info("Loaded %d reference emails.", len(references))
    return references


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(args: argparse.Namespace) -> None:
    Path("outputs").mkdir(exist_ok=True)

    # 1. Load config
    cfg = load_config("config.yaml")
    db_path = cfg["database"]["path"]
    weights = EvaluationWeights(**cfg["evaluation"])
    model_a_cfg = ModelConfig(**cfg["models"]["model_a"])
    model_b_cfg = ModelConfig(**cfg["models"]["model_b"])
    judge_cfg = ModelConfig(**cfg["judge"])

    run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    logger.info("=== Pipeline run: %s ===", run_timestamp)

    # 2. Init DB
    db.init_db(db_path)

    # 3. Load scenarios
    scenarios = load_scenarios()
    scenario_list = list(scenarios.values())

    # 4. Load references
    references = load_references()

    # 5–6. Persist scenarios and references to DB
    for s in scenario_list:
        db.upsert_scenario(s, db_path)
    for ref in references.values():
        db.upsert_reference_email(ref.scenario_id, ref.email_text, db_path)

    all_results = []

    if not args.skip_generation:
        # 7. Generate — Model A
        if not args.model_b_only:
            logger.info("--- Generation: Model A (%s) ---", model_a_cfg.name)
            results_a = generate_batch(scenario_list, model_a_cfg)
            for r in results_a:
                db.insert_generation_result(r, run_timestamp, db_path)
            all_results.extend(results_a)

        # 8. Generate — Model B
        if not args.model_a_only:
            logger.info("--- Generation: Model B (%s) ---", model_b_cfg.name)
            results_b = generate_batch(scenario_list, model_b_cfg)
            for r in results_b:
                db.insert_generation_result(r, run_timestamp, db_path)
            all_results.extend(results_b)
    else:
        logger.info("--skip-generation flag set — loading results from DB.")
        raw_rows = db.get_results_by_run(run_timestamp, db_path)
        if not raw_rows:
            logger.error("No generation results found for run %s. Exiting.", run_timestamp)
            sys.exit(1)
        from src.models import GenerationResult  # noqa: PLC0415
        all_results = [GenerationResult(**row) for row in raw_rows]

    if args.report_only:
        logger.info("--report-only flag: regenerating reports from DB results.")

    # 9. Build judge LLM and chains (Groq free tier)
    logger.info("--- Building judge chains (ChatGroq / %s) ---", judge_cfg.name)
    judge_llm = create_judge_llm(judge_cfg)
    fact_chain = build_fact_recall_chain(judge_llm)
    tone_chain = build_tone_accuracy_chain(judge_llm)
    fluency_chain = build_fluency_chain(judge_llm)

    def judge_fn(email: str, scenario: Scenario) -> dict[str, float]:
        return judge_scenario(email, scenario, fact_chain, tone_chain, fluency_chain)

    # 10. Evaluate
    logger.info("--- Evaluation Engine ---")
    all_scores = evaluate_all(
        results=all_results,
        references=references,
        scenarios=scenarios,
        judge_fn=judge_fn,
        weights=weights,
        run_timestamp=run_timestamp,
    )

    # 11. Persist scores to DB
    for score in all_scores:
        db.insert_evaluation_score(score, db_path)

    # 12. Generate reports
    logger.info("--- Generating Reports ---")
    metadata = {
        "run_timestamp": run_timestamp,
        "model_a": model_a_cfg.name,
        "model_b": model_b_cfg.name,
        "judge": judge_cfg.name,
        "scenario_count": len(scenario_list),
        "evaluation_weights": cfg["evaluation"],
    }

    df = generate_csv_report(all_scores, all_results, "outputs/evaluation_results.csv")
    generate_json_report(all_scores, all_results, metadata, "outputs/evaluation_results.json")
    generate_analysis_summary(all_scores, all_results, "outputs/analysis_summary.md", run_timestamp)

    # 13. Console summary
    print_summary_table(all_scores)
    logger.info("=== Pipeline complete. Reports in outputs/ ===")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI Email Generation Assistant — Evaluation Pipeline"
    )
    parser.add_argument(
        "--model-a-only",
        action="store_true",
        help="Only run Model A (gpt-4o) generation",
    )
    parser.add_argument(
        "--model-b-only",
        action="store_true",
        help="Only run Model B (gpt-4o-mini) generation",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip generation — re-evaluate using DB results from current run",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Regenerate reports from latest DB run (skips generation + evaluation)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
