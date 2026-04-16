"""
src/db.py
SQLite persistence layer for the AI Email Generation Assistant.
All tables are created on first run. Uses Python's built-in sqlite3 — zero infra.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from src.models import EvaluationScore, GenerationResult, Scenario

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DDL — Table definitions
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS scenarios (
    id       TEXT PRIMARY KEY,
    intent   TEXT NOT NULL,
    facts    TEXT NOT NULL,   -- JSON-serialised list[str]
    tone     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS reference_emails (
    scenario_id TEXT PRIMARY KEY,
    email_text  TEXT NOT NULL,
    FOREIGN KEY (scenario_id) REFERENCES scenarios(id)
);

CREATE TABLE IF NOT EXISTS generation_results (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    scenario_id       TEXT    NOT NULL,
    model_name        TEXT    NOT NULL,
    generated_email   TEXT    NOT NULL,
    latency_ms        REAL,
    prompt_tokens     INTEGER DEFAULT 0,
    completion_tokens INTEGER DEFAULT 0,
    status            TEXT    DEFAULT 'success',
    error_message     TEXT,
    run_timestamp     TEXT    NOT NULL,
    FOREIGN KEY (scenario_id) REFERENCES scenarios(id)
);

CREATE TABLE IF NOT EXISTS evaluation_scores (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    scenario_id             TEXT    NOT NULL,
    model_name              TEXT    NOT NULL,
    rouge_l                 REAL,
    bleu                    REAL,
    bert_score_f1           REAL,
    fact_recall             REAL,
    tone_accuracy           REAL,
    fluency_professionalism REAL,
    composite_score         REAL,
    run_timestamp           TEXT    NOT NULL,
    FOREIGN KEY (scenario_id) REFERENCES scenarios(id)
);
"""


# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------

@contextmanager
def _connect(db_path: str) -> Generator[sqlite3.Connection, None, None]:
    """Context manager that yields a connection and commits/rolls back cleanly."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def init_db(db_path: str) -> None:
    """Create all tables if they do not already exist."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with _connect(db_path) as conn:
        conn.executescript(_DDL)
    logger.info("Database initialised at %s", db_path)


def upsert_scenario(scenario: Scenario, db_path: str) -> None:
    """Insert or replace a scenario row."""
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO scenarios (id, intent, facts, tone)
            VALUES (?, ?, ?, ?)
            """,
            (scenario.id, scenario.intent, json.dumps(scenario.facts), scenario.tone),
        )


def upsert_reference_email(scenario_id: str, email_text: str, db_path: str) -> None:
    """Insert or replace a reference email row."""
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO reference_emails (scenario_id, email_text)
            VALUES (?, ?)
            """,
            (scenario_id, email_text),
        )


def insert_generation_result(result: GenerationResult, run_timestamp: str, db_path: str) -> None:
    """Persist one LLM generation output."""
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO generation_results
                (scenario_id, model_name, generated_email, latency_ms,
                 prompt_tokens, completion_tokens, status, error_message, run_timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result.scenario_id,
                result.model_name,
                result.generated_email,
                result.latency_ms,
                result.prompt_tokens,
                result.completion_tokens,
                result.status,
                result.error_message,
                run_timestamp,
            ),
        )


def insert_evaluation_score(score: EvaluationScore, db_path: str) -> None:
    """Persist one full evaluation score row."""
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO evaluation_scores
                (scenario_id, model_name, rouge_l, bleu, bert_score_f1,
                 fact_recall, tone_accuracy, fluency_professionalism,
                 composite_score, run_timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                score.scenario_id,
                score.model_name,
                score.rouge_l,
                score.bleu,
                score.bert_score_f1,
                score.fact_recall,
                score.tone_accuracy,
                score.fluency_professionalism,
                score.composite_score,
                score.run_timestamp,
            ),
        )


def get_scores_by_run(run_timestamp: str, db_path: str) -> list[dict]:
    """Return all evaluation score rows for a specific run timestamp."""
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM evaluation_scores WHERE run_timestamp = ?",
            (run_timestamp,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_all_scores(db_path: str) -> list[dict]:
    """Return every evaluation score row across all runs."""
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM evaluation_scores ORDER BY run_timestamp DESC"
        ).fetchall()
    return [dict(row) for row in rows]


def get_results_by_run(run_timestamp: str, db_path: str) -> list[dict]:
    """Return all generation result rows for a specific run timestamp."""
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM generation_results WHERE run_timestamp = ?",
            (run_timestamp,),
        ).fetchall()
    return [dict(row) for row in rows]
