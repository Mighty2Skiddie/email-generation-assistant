"""
src/llm_judge.py
LangChain ChatGroq LLM-as-Judge evaluator.

Three dedicated ChatPromptTemplate prompts, each forming an LCEL chain
(judge_prompt | ChatGroq | StrOutputParser).

All judge calls use temperature=0.0 for deterministic, reproducible scoring.
Scores are extracted with a robust multi-pattern regex parser.

Groq free-tier note: 30 requests/min — a 2-second sleep is applied between
calls to stay within rate limits for the 60-call evaluation load.
"""

from __future__ import annotations

import json
import logging
import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.runnables import RunnableSequence
from langchain_groq import ChatGroq

from src.chains import build_judge_chain
from src.models import ModelConfig, Scenario

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rate Limiting & Factory
# ---------------------------------------------------------------------------

# Global rate limiter ensuring 29 requests per minute (under the 30 free tier limit)
_GROQ_RATE_LIMITER = InMemoryRateLimiter(
    requests_per_second=29.0 / 60.0
)


def create_judge_llm(judge_config: ModelConfig) -> ChatGroq:
    """
    Instantiate a ChatGroq judge LLM.
    temperature=0.0 is enforced for deterministic scoring regardless of config.
    Uses InMemoryRateLimiter to dynamically throttle requests for the free tier.
    """
    return ChatGroq(
        model=judge_config.name,
        temperature=0.0,      # Always deterministic for evaluation
        max_tokens=judge_config.max_tokens,
        max_retries=4,        # Handle any transient API bumps gracefully
        rate_limiter=_GROQ_RATE_LIMITER,
    )


# ---------------------------------------------------------------------------
# Judge prompt templates (ChatPromptTemplate)
# ---------------------------------------------------------------------------

FACT_RECALL_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert evaluator specialising in factual accuracy assessment for "
        "professional business emails. Be strict — a fact is only PRESENT if its core "
        "information appears in the email, even if worded differently.",
    ),
    (
        "human",
        """Evaluate whether each key fact appears in the generated email.

Key Facts (JSON list): {facts}

Generated Email:
{email}

For EACH fact, respond exactly as:
FACT: <fact text> → PRESENT / ABSENT

Then on the final line respond exactly as:
SCORE: <number of PRESENT facts>/<total facts>""",
    ),
])

TONE_ACCURACY_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a professional communication expert specialising in tone analysis. "
        "You evaluate whether an email's language, style, and word choice match the requested tone.",
    ),
    (
        "human",
        """Rate how accurately this email matches the requested tone on a scale of 1 to 5.

Requested Tone: {tone}
Generated Email:
{email}

Tone Definitions:
- formal: professional, polished, no contractions, respectful distance
- casual: friendly, conversational, contractions allowed, warm
- urgent: direct, action-oriented, emphasises deadlines and consequences
- empathetic: warm, supportive, acknowledges feelings, compassionate

Scoring:
5 = Perfect tone match throughout
4 = Strong match, 1–2 minor deviations
3 = Acceptable but noticeably inconsistent
2 = Tone mismatch in major sections
1 = Completely wrong tone

Respond with ONLY these two lines:
SCORE: <1-5>
REASONING: <one sentence>""",
    ),
])

FLUENCY_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a senior editor and professional writing assessor. You evaluate "
        "business emails for grammar correctness, clarity, and overall professionalism.",
    ),
    (
        "human",
        """Rate the following email for grammar, clarity, and professionalism on a scale of 1 to 5.

Generated Email:
{email}

Scoring:
5 = Flawless grammar, crystal-clear structure, fully professional
4 = Minor grammar issues, clear, professional
3 = Some grammar errors, mostly clear
2 = Notable errors, unclear sections, unprofessional moments
1 = Poor grammar, confusing, unprofessional throughout

Respond with ONLY:
SCORE: <1-5>""",
    ),
])


# ---------------------------------------------------------------------------
# Score parsing
# ---------------------------------------------------------------------------

def parse_score_1_to_5(llm_output: str) -> float:
    """
    Extract a 1–5 numeric score from judge LLM output and normalise to [0, 1].

    Tries two patterns in order:
      1. 'SCORE: 4'  (structured format)
      2. Any standalone digit 1–5 in the text (fallback)

    Returns 0.0 and logs a warning if no score can be extracted.
    """
    # Pattern 1: explicit SCORE label
    match = re.search(r"SCORE[:\s]+([1-5](?:\.\d+)?)", llm_output, re.IGNORECASE)
    if match:
        return float(match.group(1)) / 5.0

    # Pattern 2: standalone digit
    match = re.search(r"\b([1-5])\b", llm_output)
    if match:
        logger.warning("Used fallback score parsing. Raw output: %s", llm_output[:120])
        return int(match.group(1)) / 5.0

    logger.warning("Could not parse score from judge output: %s", llm_output[:120])
    return 0.0


def parse_fact_recall_score(llm_output: str, total_facts: int) -> float:
    """
    Extract X/Y from 'SCORE: X/Y' in fact-recall judge output.
    Falls back to counting 'PRESENT' occurrences divided by total_facts.
    """
    # Pattern 1: SCORE: X/Y
    match = re.search(r"SCORE[:\s]+(\d+)\s*/\s*(\d+)", llm_output, re.IGNORECASE)
    if match:
        present = int(match.group(1))
        total = int(match.group(2))
        if total > 0:
            return min(present / total, 1.0)

    # Fallback: count PRESENT labels
    present_count = len(re.findall(r"\bPRESENT\b", llm_output, re.IGNORECASE))
    if total_facts > 0:
        logger.warning("Used PRESENT-count fallback. Found %d PRESENT.", present_count)
        return min(present_count / total_facts, 1.0)

    return 0.0


# ---------------------------------------------------------------------------
# Chain builders
# ---------------------------------------------------------------------------

def build_fact_recall_chain(judge_llm: ChatGroq) -> RunnableSequence:
    return build_judge_chain(judge_llm, FACT_RECALL_PROMPT)


def build_tone_accuracy_chain(judge_llm: ChatGroq) -> RunnableSequence:
    return build_judge_chain(judge_llm, TONE_ACCURACY_PROMPT)


def build_fluency_chain(judge_llm: ChatGroq) -> RunnableSequence:
    return build_judge_chain(judge_llm, FLUENCY_PROMPT)


# ---------------------------------------------------------------------------
# Public evaluation API
# ---------------------------------------------------------------------------

def judge_scenario(
    generated_email: str,
    scenario: Scenario,
    fact_recall_chain: RunnableSequence,
    tone_chain: RunnableSequence,
    fluency_chain: RunnableSequence,
) -> dict[str, float]:
    """
    Run all three judge chains for a single (scenario, generated_email) pair.

    Returns:
        dict with keys: fact_recall, tone_accuracy, fluency_professionalism
    """
    facts_json = json.dumps(scenario.facts, ensure_ascii=False)

    # --- Fact Recall ---
    try:
        fact_output = fact_recall_chain.invoke({
            "facts": facts_json,
            "email": generated_email,
        })
        fact_recall = parse_fact_recall_score(fact_output, total_facts=len(scenario.facts))
        logger.info("  [judge/%s] fact_recall=%.3f", scenario.id, fact_recall)
    except Exception as exc:  # noqa: BLE001
        logger.error("Fact recall judge failed for %s: %s", scenario.id, exc)
        fact_recall = 0.0

    # --- Tone Accuracy ---
    try:
        tone_output = tone_chain.invoke({
            "tone": scenario.tone,
            "email": generated_email,
        })
        tone_accuracy = parse_score_1_to_5(tone_output)
        logger.info("  [judge/%s] tone_accuracy=%.3f", scenario.id, tone_accuracy)
    except Exception as exc:  # noqa: BLE001
        logger.error("Tone accuracy judge failed for %s: %s", scenario.id, exc)
        tone_accuracy = 0.0

    # --- Fluency & Professionalism ---
    try:
        fluency_output = fluency_chain.invoke({"email": generated_email})
        fluency = parse_score_1_to_5(fluency_output)
        logger.info("  [judge/%s] fluency=%.3f", scenario.id, fluency)
    except Exception as exc:  # noqa: BLE001
        logger.error("Fluency judge failed for %s: %s", scenario.id, exc)
        fluency = 0.0

    return {
        "fact_recall": fact_recall,
        "tone_accuracy": tone_accuracy,
        "fluency_professionalism": fluency,
    }
