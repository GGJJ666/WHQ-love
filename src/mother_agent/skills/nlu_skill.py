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

        context = context or {}
        product_term_matches = self._extract_product_terms(prompt, context)
        product_spans = [span for _, span in product_term_matches]

        # Naive entity extraction: capitalised tokens that are not sentence-start words
        entities: list[str] = []
        seen_entities: set[str] = set()
        for term, _ in product_term_matches:
            if term in seen_entities:
                continue
            entities.append(term)
            seen_entities.add(term)
        for match in re.finditer(r"\b[A-Z][a-z]+\b", prompt):
            if any(self._spans_overlap(match.span(), span) for span in product_spans):
                continue
            token = match.group(0)
            if token in {"I", "The", "A", "An", "This", "That"}:
                continue
            if token in seen_entities:
                continue
            entities.append(token)
            seen_entities.add(token)

        output = f"Intent: {intent} | Entities: {entities}"
        return SkillResult(
            skill_name=self.name,
            output=output,
            data={"intent": intent, "entities": entities},
        )

    def _extract_product_terms(
        self,
        prompt: str,
        context: dict[str, Any],
    ) -> list[tuple[str, tuple[int, int]]]:
        terms = self._get_product_terms(context)
        matches: list[tuple[str, tuple[int, int]]] = []
        matched_spans: list[tuple[int, int]] = []

        # Match longer terms first so a shorter name like "Love" does not
        # consume part of a larger product name such as "WHQ Love".
        for term in sorted(terms, key=len, reverse=True):
            pattern = re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", re.IGNORECASE)
            for match in pattern.finditer(prompt):
                if any(self._spans_overlap(match.span(), span) for span in matched_spans):
                    continue
                matched_spans.append(match.span())
                # Keep the configured canonical spelling in extracted entities
                # even when the prompt uses different casing.
                matches.append((term, match.span()))

        return sorted(matches, key=lambda item: item[1][0])

    def _get_product_terms(self, context: dict[str, Any]) -> list[str]:
        """Return product terms with explicit-call context taking precedence.

        Direct ``product_terms`` in the call context override any
        ``agent_metadata["product_terms"]`` injected by the agent runtime.
        """
        terms = context.get("product_terms")
        if terms is None:
            terms = context.get("agent_metadata", {}).get("product_terms")
        if isinstance(terms, str):
            terms = [terms]
        if not isinstance(terms, list):
            return []
        return [term.strip() for term in terms if isinstance(term, str) and term.strip()]

    @staticmethod
    def _spans_overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
        return left[0] < right[1] and right[0] < left[1]
