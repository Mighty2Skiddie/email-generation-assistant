"""
tests/test_prompt_engine.py
Unit tests for the LangChain prompt construction engine.

Validates ChatPromptTemplate structure, input variables, few-shot injection,
and message role correctness — all without calling any LLM APIs.
"""

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

from src.models import Scenario
from src.prompt_engine import (
    build_few_shot_prompt,
    build_generation_prompt,
    format_scenario_inputs,
    get_few_shot_examples,
    get_system_prompt,
)


class TestGetFewShotExamples:
    def test_returns_list(self):
        examples = get_few_shot_examples()
        assert isinstance(examples, list)

    def test_at_least_two_examples(self):
        examples = get_few_shot_examples()
        assert len(examples) >= 2

    def test_example_has_required_keys(self):
        for ex in get_few_shot_examples():
            assert "intent" in ex
            assert "facts" in ex
            assert "tone" in ex
            assert "email" in ex

    def test_example_tone_is_valid(self):
        valid_tones = {"formal", "casual", "urgent", "empathetic"}
        for ex in get_few_shot_examples():
            assert ex["tone"] in valid_tones

    def test_examples_cover_multiple_tones(self):
        tones = {ex["tone"] for ex in get_few_shot_examples()}
        assert len(tones) >= 2, "Examples should cover at least 2 different tones"


class TestGetSystemPrompt:
    def test_returns_system_template(self):
        sp = get_system_prompt()
        # Should produce a SystemMessage when formatted
        msgs = sp.format_messages()
        assert len(msgs) == 1
        assert isinstance(msgs[0], SystemMessage)

    def test_system_message_contains_persona(self):
        sp = get_system_prompt()
        content = sp.format_messages()[0].content
        assert "professional" in content.lower()
        assert "facts" in content.lower()
        assert "tone" in content.lower()


class TestBuildFewShotPrompt:
    def test_returns_few_shot_template(self):
        fsp = build_few_shot_prompt()
        assert isinstance(fsp, FewShotChatMessagePromptTemplate)

    def test_formats_to_human_ai_pairs(self):
        fsp = build_few_shot_prompt()
        msgs = fsp.format_messages()
        # Each example = 1 HumanMessage + 1 AIMessage
        assert len(msgs) == len(get_few_shot_examples()) * 2
        for i in range(0, len(msgs), 2):
            assert isinstance(msgs[i], HumanMessage)
            assert isinstance(msgs[i + 1], AIMessage)

    def test_human_message_contains_intent(self):
        fsp = build_few_shot_prompt()
        msgs = fsp.format_messages()
        human_msgs = [m for m in msgs if isinstance(m, HumanMessage)]
        for hw in human_msgs:
            assert "Intent:" in hw.content

    def test_ai_message_non_empty(self):
        fsp = build_few_shot_prompt()
        msgs = fsp.format_messages()
        ai_msgs = [m for m in msgs if isinstance(m, AIMessage)]
        for ai in ai_msgs:
            assert len(ai.content.strip()) > 20


class TestBuildGenerationPrompt:
    def test_returns_chat_prompt_template(self):
        prompt = build_generation_prompt()
        assert isinstance(prompt, ChatPromptTemplate)

    def test_input_variables_correct(self):
        prompt = build_generation_prompt()
        assert set(prompt.input_variables) == {"intent", "facts", "tone"}

    def test_formats_correctly_with_all_inputs(self):
        prompt = build_generation_prompt()
        msgs = prompt.format_messages(
            intent="Request a meeting",
            facts=json.dumps(["Available Monday 3PM", "Agenda: Q2 planning"]),
            tone="formal",
        )
        assert len(msgs) > 0
        # First message must be SystemMessage
        assert isinstance(msgs[0], SystemMessage)
        # Last message must be the HumanMessage with the actual task
        assert isinstance(msgs[-1], HumanMessage)
        assert "Request a meeting" in msgs[-1].content

    def test_system_message_is_first(self):
        prompt = build_generation_prompt()
        msgs = prompt.format_messages(
            intent="Test", facts='["fact1"]', tone="casual"
        )
        assert isinstance(msgs[0], SystemMessage)

    def test_few_shot_pairs_present_in_middle(self):
        prompt = build_generation_prompt()
        msgs = prompt.format_messages(
            intent="Test intent", facts='["fact"]', tone="urgent"
        )
        types = [type(m).__name__ for m in msgs]
        assert "HumanMessage" in types
        assert "AIMessage" in types

    def test_task_human_message_contains_tone(self):
        prompt = build_generation_prompt()
        msgs = prompt.format_messages(
            intent="Send reminder", facts='["Meeting at 10 AM"]', tone="empathetic"
        )
        last_human = msgs[-1]
        assert "empathetic" in last_human.content


class TestFormatScenarioInputs:
    def _make_scenario(self):
        return Scenario(
            id="S01",
            intent="Follow up after meeting",
            facts=["Met on Tuesday", "Send proposal by Friday"],
            tone="formal",
        )

    def test_returns_dict(self):
        s = self._make_scenario()
        result = format_scenario_inputs(s)
        assert isinstance(result, dict)

    def test_has_required_keys(self):
        s = self._make_scenario()
        result = format_scenario_inputs(s)
        assert "intent" in result
        assert "facts" in result
        assert "tone" in result

    def test_facts_is_json_string(self):
        s = self._make_scenario()
        result = format_scenario_inputs(s)
        parsed = json.loads(result["facts"])
        assert isinstance(parsed, list)
        assert len(parsed) == 2

    def test_intent_matches_scenario(self):
        s = self._make_scenario()
        result = format_scenario_inputs(s)
        assert result["intent"] == s.intent

    def test_tone_matches_scenario(self):
        s = self._make_scenario()
        result = format_scenario_inputs(s)
        assert result["tone"] == s.tone
