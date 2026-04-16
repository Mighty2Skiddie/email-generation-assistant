"""
src/generation_client.py
LangChain ChatOpenAI generation client.

Wraps both Model A (gpt-4o) and Model B (gpt-4o-mini) with:
  - TokenUsageCallback  — captures prompt/completion token counts
  - generate_email()    — single scenario with full metadata capture
  - generate_batch()    — sequential loop with progress logging

Retry and timeout are handled natively by ChatOpenAI(max_retries, timeout).
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI

from src.chains import build_generation_chain
from src.models import GenerationResult, ModelConfig, Scenario
from src.prompt_engine import format_scenario_inputs

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Callback — capture token usage
# ---------------------------------------------------------------------------

class TokenUsageCallback(BaseCallbackHandler):
    """Records prompt_tokens and completion_tokens from the LLM response metadata."""

    def __init__(self) -> None:
        super().__init__()
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:  # noqa: ARG002
        usage = (response.llm_output or {}).get("token_usage", {})
        self.prompt_tokens = usage.get("prompt_tokens", 0)
        self.completion_tokens = usage.get("completion_tokens", 0)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_llm(model_config: ModelConfig) -> ChatOpenAI:
    """
    Instantiate a ChatOpenAI from a ModelConfig.

    Built-in LangChain features used:
      - max_retries=3   → exponential backoff on rate-limit / transient errors
      - timeout=30      → hard 30-second per-call timeout
    """
    return ChatOpenAI(
        model=model_config.name,
        temperature=model_config.temperature,
        max_tokens=model_config.max_tokens,
        timeout=30,
        max_retries=3,
    )


# ---------------------------------------------------------------------------
# Core generation functions
# ---------------------------------------------------------------------------

def generate_email(
    scenario: Scenario,
    chain: RunnableSequence,
    model_name: str,
) -> GenerationResult:
    """
    Generate a single email for one scenario.

    Args:
        scenario:   Validated Pydantic Scenario object.
        chain:      Pre-built LCEL generation chain (prompt | llm | parser).
        model_name: Human-readable model identifier for metadata.

    Returns:
        GenerationResult with generated text, latency, and token counts.
    """
    callback = TokenUsageCallback()
    inputs = format_scenario_inputs(scenario)

    start = time.perf_counter()
    try:
        email_text: str = chain.invoke(
            inputs,
            config={"callbacks": [callback]},
        )
        latency_ms = (time.perf_counter() - start) * 1000

        if not email_text or not email_text.strip():
            logger.warning("Empty response for scenario %s — retrying once.", scenario.id)
            email_text = chain.invoke(inputs, config={"callbacks": [callback]})
            latency_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "[%s] %s | %.0f ms | %d+%d tokens",
            model_name,
            scenario.id,
            latency_ms,
            callback.prompt_tokens,
            callback.completion_tokens,
        )

        return GenerationResult(
            scenario_id=scenario.id,
            model_name=model_name,
            generated_email=email_text.strip(),
            latency_ms=round(latency_ms, 2),
            prompt_tokens=callback.prompt_tokens,
            completion_tokens=callback.completion_tokens,
            status="success",
        )

    except Exception as exc:  # noqa: BLE001
        latency_ms = (time.perf_counter() - start) * 1000
        logger.error("Generation FAILED for %s / %s: %s", model_name, scenario.id, exc)
        return GenerationResult(
            scenario_id=scenario.id,
            model_name=model_name,
            generated_email="",
            latency_ms=round(latency_ms, 2),
            status="failed",
            error_message=str(exc),
        )


def generate_batch(
    scenarios: list[Scenario],
    model_config: ModelConfig,
) -> list[GenerationResult]:
    """
    Generate emails for all scenarios using one model.

    Args:
        scenarios:    List of validated Scenario objects.
        model_config: ModelConfig (provider, name, temperature, max_tokens).

    Returns:
        List of GenerationResult, one per scenario.
    """
    llm = create_llm(model_config)
    chain = build_generation_chain(llm)
    results: list[GenerationResult] = []

    logger.info("=== Generating with %s (%d scenarios) ===", model_config.name, len(scenarios))

    for i, scenario in enumerate(scenarios, start=1):
        logger.info("  [%d/%d] Scenario %s ...", i, len(scenarios), scenario.id)
        result = generate_email(scenario, chain, model_config.name)
        results.append(result)

    successes = sum(1 for r in results if r.status == "success")
    logger.info(
        "=== %s complete: %d/%d succeeded ===",
        model_config.name,
        successes,
        len(scenarios),
    )
    return results
