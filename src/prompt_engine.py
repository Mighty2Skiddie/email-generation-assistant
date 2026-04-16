"""
src/prompt_engine.py
LangChain-based prompt construction engine.

Advanced technique: Few-Shot Prompting + Role-Playing
  - SystemMessagePromptTemplate  → assigns expert persona
  - FewShotChatMessagePromptTemplate → injects 3 curated gold-standard examples
  - HumanMessagePromptTemplate   → carries the actual scenario variables

All prompts are built as LangChain ChatPromptTemplate objects for seamless
LCEL composition (prompt | llm | parser).
"""

from __future__ import annotations

import json
import logging

from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

from src.models import Scenario

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System persona
# ---------------------------------------------------------------------------

_SYSTEM_INSTRUCTION = """\
You are an expert professional communication specialist with 20 years of \
experience drafting high-stakes business correspondence. You always:
- Include ALL provided facts verbatim or semantically — never omit a fact
- Match the exact tone requested without deviation
- Write concise, clear, professional emails with no filler sentences
- Use appropriate greetings and sign-offs that match the tone
- Never include unfilled placeholder text such as [Your Name] or [Date]
- Output ONLY the email text — no preamble, no explanation\
"""

# ---------------------------------------------------------------------------
# Few-shot examples  (3 examples covering formal, casual, urgent)
# ---------------------------------------------------------------------------

_FEW_SHOT_EXAMPLES: list[dict] = [
    {
        "intent": "Follow up on a contract negotiation",
        "facts": json.dumps([
            "Met on Monday March 3rd",
            "Discussed revised payment terms: Net-45 instead of Net-30",
            "Legal team to review contract draft by Friday",
            "Target signing date is March 20th",
        ]),
        "tone": "formal",
        "email": (
            "Subject: Follow-Up — Contract Negotiation | Next Steps\n\n"
            "Dear [Recipient],\n\n"
            "Thank you for our productive meeting on Monday, March 3rd. "
            "I wanted to confirm the key points we discussed regarding the revised payment terms.\n\n"
            "As agreed, we will move to Net-45 payment terms in place of the original Net-30. "
            "Our legal team is currently reviewing the updated contract draft and will complete "
            "their review by Friday. We remain on track for a target signing date of March 20th.\n\n"
            "Please do not hesitate to reach out should you require any clarification in the interim.\n\n"
            "Best regards,\n"
            "[Sender Name]\n"
            "[Title] | [Company]"
        ),
    },
    {
        "intent": "Announce a team-building event",
        "facts": json.dumps([
            "Event is a bowling night on Thursday at 6 PM",
            "Location: Lucky Strike Lanes, Downtown",
            "Shoes and two games included, company-paid",
            "RSVP by Wednesday to secure enough lanes",
        ]),
        "tone": "casual",
        "email": (
            "Subject: Team Bowling Night — Thursday 6 PM! 🎳\n\n"
            "Hey everyone!\n\n"
            "Get ready for some friendly competition — we're heading out for a bowling night "
            "this Thursday at 6 PM at Lucky Strike Lanes downtown!\n\n"
            "Shoe rental and two games are fully covered by the company, so all you need to bring "
            "is your competitive spirit. To make sure we book enough lanes, please RSVP by Wednesday.\n\n"
            "See you on the lanes!\n\n"
            "[Sender Name]"
        ),
    },
    {
        "intent": "Alert the team about a security breach requiring immediate password resets",
        "facts": json.dumps([
            "Unauthorized access detected in the CRM system at 11 PM last night",
            "All user passwords must be reset immediately",
            "IT support is available 24/7 at ext. 4400",
            "Do not access company systems until password reset is confirmed",
        ]),
        "tone": "urgent",
        "email": (
            "Subject: URGENT — Immediate Password Reset Required | Security Alert\n\n"
            "All Staff,\n\n"
            "IMPORTANT: We detected unauthorized access in our CRM system last night at 11 PM. "
            "As a precautionary measure, all user passwords must be reset immediately.\n\n"
            "ACTION REQUIRED:\n"
            "- Do NOT access any company systems until your password reset is confirmed\n"
            "- Reset your password now using the link sent to your registered email\n"
            "- Contact IT Support at ext. 4400 (available 24/7) if you need assistance\n\n"
            "Your immediate action is critical to protecting our systems and data. "
            "Please complete your reset within the next 30 minutes.\n\n"
            "[Sender Name]\n"
            "IT Security Team"
        ),
    },
]

# ---------------------------------------------------------------------------
# Example prompt template (one human+ai pair per example)
# ---------------------------------------------------------------------------

_EXAMPLE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("human", "Intent: {intent}\nFacts: {facts}\nTone: {tone}\n\nWrite the professional email."),
        ("ai", "{email}"),
    ]
)

# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------

def get_few_shot_examples() -> list[dict]:
    """Return the 3 curated few-shot example dicts."""
    return _FEW_SHOT_EXAMPLES


def build_few_shot_prompt() -> FewShotChatMessagePromptTemplate:
    """Assemble the FewShotChatMessagePromptTemplate from the curated examples."""
    return FewShotChatMessagePromptTemplate(
        example_prompt=_EXAMPLE_PROMPT,
        examples=_FEW_SHOT_EXAMPLES,
    )


def get_system_prompt() -> SystemMessagePromptTemplate:
    """Return the role-play system message template."""
    return SystemMessagePromptTemplate.from_template(_SYSTEM_INSTRUCTION)


def build_generation_prompt() -> ChatPromptTemplate:
    """
    Build the full generation ChatPromptTemplate:
        System (role-play persona)
        + Few-Shot block (3 examples)
        + Human (actual task variables: intent, facts, tone)

    Input variables expected by the assembled prompt:
        intent  str   — core purpose of the email
        facts   str   — JSON-serialised list of facts
        tone    str   — one of formal / casual / urgent / empathetic
    """
    few_shot = build_few_shot_prompt()
    prompt = ChatPromptTemplate.from_messages(
        [
            get_system_prompt(),
            few_shot,
            ("human", "Intent: {intent}\nFacts: {facts}\nTone: {tone}\n\nWrite the professional email."),
        ]
    )
    logger.debug("Generation prompt built. Input variables: %s", prompt.input_variables)
    return prompt


def format_scenario_inputs(scenario: Scenario) -> dict[str, str]:
    """Convert a Pydantic Scenario into the dict expected by the prompt template."""
    return {
        "intent": scenario.intent,
        "facts": json.dumps(scenario.facts, ensure_ascii=False),
        "tone": scenario.tone,
    }
