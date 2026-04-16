"""
src/chains.py
LCEL (LangChain Expression Language) chain factory.

All chains follow the same declarative pattern:
    prompt | llm | parser

This module is the single source of truth for chain construction.
Both generation and judge chains are built here to ensure consistent
interface across all LLM calls.
"""

from __future__ import annotations

import logging

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence

from src.prompt_engine import build_generation_prompt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Generation chain
# ---------------------------------------------------------------------------

def build_generation_chain(llm: BaseChatModel) -> RunnableSequence:
    """
    Email generation chain.

    Pipeline:
        ChatPromptTemplate (few-shot + role-play)
        | ChatOpenAI (gpt-4o or gpt-4o-mini)
        | StrOutputParser  →  plain string email text

    Args:
        llm: A LangChain ChatOpenAI (or any BaseChatModel) instance.

    Returns:
        RunnableSequence that accepts dict(intent, facts, tone) and returns str.
    """
    prompt = build_generation_prompt()
    chain: RunnableSequence = prompt | llm | StrOutputParser()
    logger.debug("Generation chain built for model: %s", getattr(llm, "model_name", "unknown"))
    return chain


# ---------------------------------------------------------------------------
# Judge chains
# ---------------------------------------------------------------------------

def build_judge_chain(
    llm: BaseChatModel,
    judge_prompt: ChatPromptTemplate,
) -> RunnableSequence:
    """
    Generic LLM-as-Judge evaluation chain.

    Pipeline:
        judge_prompt (ChatPromptTemplate)
        | ChatGroq (llama-3.3-70b-versatile, temperature=0)
        | StrOutputParser  →  raw judge response string

    The caller is responsible for parsing the score from the returned string
    using `llm_judge.parse_score()`.

    Args:
        llm:          A LangChain ChatGroq (or any BaseChatModel) judge instance.
        judge_prompt: The specific ChatPromptTemplate for this judge task.

    Returns:
        RunnableSequence that accepts the judge prompt's input_variables dict
        and returns the raw LLM response string.
    """
    chain: RunnableSequence = judge_prompt | llm | StrOutputParser()
    return chain
