"""Mother Agent — central coordinator that spawns and manages child agents."""

from __future__ import annotations

import threading
import time
from typing import Any

from .backbone.base_backbone import BaseBackbone, BackboneOutput, MockBackbone
from .config import AgentConfig
from .skills.base_skill import BaseSkill, SkillResult
from .skills.nlu_skill import NLUSkill
from .skills.planning_skill import PlanningSkill
from .skills.tool_skill import ToolSkill


class MotherAgent:
    """Central agent that owns the shared backbone and spawns child agents.

    The :class:`MotherAgent` is the single owner of the heavy shared model
    weights (represented here by a :class:`~backbone.BaseBackbone` instance).
    It maintains a registry of pluggable skills and a registry of spawned
    child agents.

    Child agents are created via :meth:`spawn_child` which accepts a
    lightweight :class:`~config.AgentConfig` override and returns a fully
    initialised :class:`~child_agent.ChildAgent`.

    Thread safety
    ~~~~~~~~~~~~~
    The skill registry and child registry are protected by a reentrant lock,
    making it safe to register skills and spawn / retrieve children from
    multiple threads simultaneously.

    Args:
        config: Configuration for the mother agent itself.
        backbone: Shared backbone model.  Defaults to :class:`MockBackbone`.
    """

    def __init__(
        self,
        config: AgentConfig | None = None,
        backbone: BaseBackbone | None = None,
    ) -> None:
        self.config: AgentConfig = config or AgentConfig(
            agent_id="mother",
            role="mother",
        )
        self.backbone: BaseBackbone = backbone or MockBackbone()
        self._skills: dict[str, BaseSkill] = {}
        self._children: dict[str, "ChildAgent"] = {}
        self._lock = threading.RLock()

        # Register built-in skills
        self._register_default_skills()

    # ------------------------------------------------------------------
    # Skill registry
    # ------------------------------------------------------------------

    def _register_default_skills(self) -> None:
        for skill in [NLUSkill(), PlanningSkill(), ToolSkill()]:
            self.register_skill(skill)

    def register_skill(self, skill: BaseSkill) -> None:
        """Add *skill* to the shared skill registry.

        Args:
            skill: Any :class:`~skills.BaseSkill` instance.
        """
        with self._lock:
            self._skills[skill.name] = skill

    def unregister_skill(self, skill_name: str) -> None:
        """Remove the skill named *skill_name* from the registry.

        Args:
            skill_name: Name as returned by :attr:`BaseSkill.name`.

        Raises:
            KeyError: If *skill_name* is not registered.
        """
        with self._lock:
            if skill_name not in self._skills:
                raise KeyError(f"Skill '{skill_name}' is not registered.")
            del self._skills[skill_name]

    @property
    def available_skills(self) -> list[str]:
        """Names of all registered skills."""
        with self._lock:
            return list(self._skills.keys())

    def get_skill(self, skill_name: str) -> BaseSkill:
        """Retrieve a registered skill by name.

        Args:
            skill_name: Skill identifier.

        Raises:
            KeyError: If *skill_name* is not registered.
        """
        with self._lock:
            if skill_name not in self._skills:
                raise KeyError(f"Skill '{skill_name}' not found.")
            return self._skills[skill_name]

    # ------------------------------------------------------------------
    # Child agent management
    # ------------------------------------------------------------------

    def spawn_child(self, child_config: AgentConfig) -> "ChildAgent":
        """Create and register a new :class:`ChildAgent`.

        The child receives read-only references to this mother's backbone and
        skill registry.  It may apply an optional adapter layer on top.

        Args:
            child_config: Lightweight config describing the child's
                specialisation.

        Returns:
            A fully initialised :class:`ChildAgent`.

        Raises:
            ValueError: If an agent with ``child_config.agent_id`` is already
                registered.
        """
        from .child_agent import ChildAgent  # avoid circular import at module level

        with self._lock:
            if child_config.agent_id in self._children:
                raise ValueError(
                    f"A child agent with id '{child_config.agent_id}' already exists. "
                    "Use a unique agent_id."
                )
            child = ChildAgent(
                config=child_config,
                mother=self,
            )
            self._children[child_config.agent_id] = child
            return child

    def get_child(self, agent_id: str) -> "ChildAgent":
        """Retrieve a previously spawned child agent by its ID.

        Args:
            agent_id: Identifier given at :meth:`spawn_child` time.

        Raises:
            KeyError: If no child with *agent_id* is registered.
        """
        with self._lock:
            if agent_id not in self._children:
                raise KeyError(f"No child agent with id '{agent_id}'.")
            return self._children[agent_id]

    @property
    def children(self) -> dict[str, "ChildAgent"]:
        """Snapshot of all currently registered child agents."""
        with self._lock:
            return dict(self._children)

    def deregister_child(self, agent_id: str) -> None:
        """Remove a child agent from the registry.

        Args:
            agent_id: ID of the child to remove.

        Raises:
            KeyError: If *agent_id* is not registered.
        """
        with self._lock:
            if agent_id not in self._children:
                raise KeyError(f"No child agent with id '{agent_id}'.")
            del self._children[agent_id]

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def run(
        self,
        prompt: str,
        skill_name: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> BackboneOutput | SkillResult:
        """Execute a prompt with the mother agent directly.

        If *skill_name* is provided the corresponding skill is invoked
        instead of the backbone.

        Args:
            prompt: User input or sub-task description.
            skill_name: Optional skill to route the request to.
            context: Optional context dictionary.

        Returns:
            :class:`~backbone.BackboneOutput` when using the backbone, or
            :class:`~skills.SkillResult` when using a skill.
        """
        if skill_name is not None:
            skill = self.get_skill(skill_name)
            skill_context = {
                "agent_metadata": dict(self.config.metadata),
                **(context or {}),
            }
            return skill.execute(prompt, skill_context)

        start = time.perf_counter()
        result = self.backbone.generate(prompt, **self.config.generation_kwargs)
        result.latency_ms = (time.perf_counter() - start) * 1000
        return result

    # ------------------------------------------------------------------
    # Broadcast updates to children
    # ------------------------------------------------------------------

    def broadcast_skill_update(self, skill: BaseSkill) -> None:
        """Re-register *skill* so all children pick up the latest version.

        Because children hold a reference to the mother's skill registry,
        updating the registry is sufficient — no per-child update is needed.

        Args:
            skill: Updated skill instance to push to all children.
        """
        self.register_skill(skill)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"MotherAgent(id={self.config.agent_id!r}, "
            f"backbone={self.backbone.model_name!r}, "
            f"children={len(self._children)}, "
            f"skills={self.available_skills})"
        )
