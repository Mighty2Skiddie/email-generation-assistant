"""
tests/test_chains.py
Unit tests for LCEL chain construction.

Uses LangChain's FakeListChatModel to test chain structure and invocation
without making any real API calls — zero cost, fully offline.
"""

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence

from src.chains import build_generation_chain, build_judge_chain


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_llm(responses: list[str]) -> FakeListChatModel:
    """Create a deterministic fake LLM that returns preset responses."""
    return FakeListChatModel(responses=responses)


_JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a judge."),
    ("human", "Score this email: {email}\nSCORE: [1-5]"),
])


# ---------------------------------------------------------------------------
# Generation chain
# ---------------------------------------------------------------------------

class TestBuildGenerationChain:
    def test_returns_runnable_sequence(self):
        llm = _fake_llm(["Test email response"])
        chain = build_generation_chain(llm)
        assert isinstance(chain, RunnableSequence)

    def test_chain_returns_string(self):
        expected = "Dear Client, thank you for your time."
        llm = _fake_llm([expected])
        chain = build_generation_chain(llm)
        result = chain.invoke({
            "intent": "Follow up after meeting",
            "facts": '["Met Monday", "Proposal due Friday"]',
            "tone": "formal",
        })
        assert isinstance(result, str)
        assert len(result) > 0

    def test_chain_returns_expected_output(self):
        expected = "Hello, this is a test email."
        llm = _fake_llm([expected])
        chain = build_generation_chain(llm)
        result = chain.invoke({
            "intent": "Test intent",
            "facts": '["fact1"]',
            "tone": "casual",
        })
        assert result == expected

    def test_chain_accepts_all_tones(self):
        for tone in ("formal", "casual", "urgent", "empathetic"):
            llm = _fake_llm([f"Email for {tone} tone."])
            chain = build_generation_chain(llm)
            result = chain.invoke({
                "intent": "Test",
                "facts": '["fact"]',
                "tone": tone,
            })
            assert isinstance(result, str)

    def test_different_llms_produce_different_chains(self):
        llm_a = _fake_llm(["Response from A"])
        llm_b = _fake_llm(["Response from B"])
        chain_a = build_generation_chain(llm_a)
        chain_b = build_generation_chain(llm_b)
        result_a = chain_a.invoke({"intent": "X", "facts": '[]', "tone": "formal"})
        result_b = chain_b.invoke({"intent": "X", "facts": '[]', "tone": "formal"})
        assert result_a != result_b


# ---------------------------------------------------------------------------
# Judge chain
# ---------------------------------------------------------------------------

class TestBuildJudgeChain:
    def test_returns_runnable_sequence(self):
        llm = _fake_llm(["SCORE: 4\nREASONING: Good tone match."])
        chain = build_judge_chain(llm, _JUDGE_PROMPT)
        assert isinstance(chain, RunnableSequence)

    def test_chain_returns_string(self):
        response = "SCORE: 5\nREASONING: Perfect."
        llm = _fake_llm([response])
        chain = build_judge_chain(llm, _JUDGE_PROMPT)
        result = chain.invoke({"email": "Dear team, please act now."})
        assert isinstance(result, str)

    def test_chain_returns_expected_judge_output(self):
        expected = "SCORE: 3"
        llm = _fake_llm([expected])
        chain = build_judge_chain(llm, _JUDGE_PROMPT)
        result = chain.invoke({"email": "Hi there."})
        assert result == expected

    def test_judge_chain_with_different_prompts(self):
        prompt_a = ChatPromptTemplate.from_messages([("human", "Evaluate: {email}")])
        prompt_b = ChatPromptTemplate.from_messages([("human", "Rate: {email}")])
        llm_a = _fake_llm(["SCORE: 4"])
        llm_b = _fake_llm(["SCORE: 2"])
        chain_a = build_judge_chain(llm_a, prompt_a)
        chain_b = build_judge_chain(llm_b, prompt_b)
        res_a = chain_a.invoke({"email": "Some email text."})
        res_b = chain_b.invoke({"email": "Some email text."})
        assert res_a == "SCORE: 4"
        assert res_b == "SCORE: 2"
