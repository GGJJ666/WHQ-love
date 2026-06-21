"""Planning skill module — decomposes goals into ordered sub-tasks."""

from __future__ import annotations

from typing import Any

from .base_skill import BaseSkill, SkillResult


class PlanningSkill(BaseSkill):
    """Decomposes a high-level goal into an ordered list of sub-tasks.

    The reference implementation uses a simple sentence-splitting heuristic.
    Replace with a chain-of-thought or tree-of-thought LLM call in production.
    """

    @property
    def name(self) -> str:
        return "planning"

    @property
    def description(self) -> str:
        return "Breaks down a complex goal into a sequence of actionable sub-tasks."

    def execute(self, prompt: str, context: dict[str, Any] | None = None) -> SkillResult:
        ctx = context or {}
        max_steps: int = ctx.get("max_steps", 5)

        # Heuristic plan generation: split on conjunctions / punctuation
        import re

        raw_steps = re.split(r"[,;]|\band\b|\bthen\b|\bnext\b|\bafter that\b", prompt, flags=re.I)
        steps = [s.strip().capitalize() for s in raw_steps if s.strip()][:max_steps]

        if not steps:
            steps = [f"Analyze the goal: {prompt[:60]}", "Execute the primary task", "Verify results"]

        numbered = "\n".join(f"{i+1}. {step}" for i, step in enumerate(steps))
        output = f"Plan ({len(steps)} steps):\n{numbered}"

        return SkillResult(
            skill_name=self.name,
            output=output,
            data={"steps": steps, "num_steps": len(steps)},
        )
