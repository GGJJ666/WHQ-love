"""Abstract base class for all skill modules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SkillResult:
    """Container for the output of a skill execution.

    Attributes:
        skill_name: Name of the skill that produced this result.
        output: Primary textual output of the skill.
        data: Structured data payload (tool results, parsed entities, …).
        success: Whether the skill executed successfully.
        error: Error message if ``success`` is ``False``.
    """

    skill_name: str
    output: str
    data: dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: str = ""


class BaseSkill(ABC):
    """Abstract base class for a pluggable skill module.

    Every skill receives a *prompt* (the raw user/agent input) and an optional
    *context* dictionary that carries additional state (conversation history,
    retrieved documents, etc.).

    Subclasses implement :meth:`execute` to perform the actual skill logic.
    The :attr:`name` property uniquely identifies the skill within the registry.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this skill."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of what this skill does."""

    @abstractmethod
    def execute(self, prompt: str, context: dict[str, Any] | None = None) -> SkillResult:
        """Execute the skill on the given *prompt*.

        Args:
            prompt: Input text (user query or agent sub-task).
            context: Optional dictionary with additional context.

        Returns:
            :class:`SkillResult` describing the outcome.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
