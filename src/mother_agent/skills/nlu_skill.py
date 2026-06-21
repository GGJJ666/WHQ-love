"""Natural Language Understanding skill module."""

from __future__ import annotations

import re
from typing import Any

from .base_skill import BaseSkill, SkillResult


class NLUSkill(BaseSkill):
    """Performs intent detection and entity extraction on user input.

    In a production system this would delegate to a fine-tuned NLU model
    or a structured extraction API.  This reference implementation uses
    simple rule-based heuristics so that the framework is fully functional
    without external dependencies.
    """

    _INTENT_PATTERNS: list[tuple[str, str]] = [
        (r"\b(search|find|look up|query)\b", "search"),
        (r"\b(plan|schedule|organize|arrange)\b", "plan"),
        (r"\b(calculate|compute|solve|math)\b", "calculate"),
        (r"\b(summarize|summarise|brief|tldr)\b", "summarize"),
        (r"\b(translate|convert)\b", "translate"),
        (r"\b(help|assist|support)\b", "help"),
        (r"\b(create|generate|write|make)\b", "create"),
        (r"\b(explain|describe|what is|what are)\b", "explain"),
    ]

    @property
    def name(self) -> str:
        return "nlu"

    @property
    def description(self) -> str:
        return "Detects intent and extracts entities from natural language input."

    def execute(self, prompt: str, context: dict[str, Any] | None = None) -> SkillResult:
        lower = prompt.lower()
        intent = "unknown"
        for pattern, label in self._INTENT_PATTERNS:
            if re.search(pattern, lower):
                intent = label
                break

        # Naive entity extraction: capitalised tokens that are not sentence-start words
        tokens = re.findall(r"\b[A-Z][a-z]+\b", prompt)
        entities = list({t for t in tokens if t not in {"I", "The", "A", "An", "This", "That"}})

        output = f"Intent: {intent} | Entities: {entities}"
        return SkillResult(
            skill_name=self.name,
            output=output,
            data={"intent": intent, "entities": entities},
        )
