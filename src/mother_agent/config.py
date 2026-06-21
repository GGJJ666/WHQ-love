"""Configuration management for Mother and Child agents."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentConfig:
    """Immutable configuration bundle for any agent in the hierarchy.

    Attributes:
        agent_id: Unique identifier string for this agent.
        role: High-level role descriptor (e.g. ``"mother"``, ``"child"``).
        domain: Optional domain specialisation (e.g. ``"medical"``,
            ``"finance"``).
        task: Optional task type (e.g. ``"qa"``, ``"summarisation"``).
        enabled_skills: List of skill names that this agent should load.
            An empty list means *all* available skills are enabled.
        adapter_type: Which adapter to attach to the backbone.
            One of ``"prompt"``, ``"lora"``, or ``None`` (no adapter).
        adapter_params: Parameters forwarded to the adapter constructor.
        max_retries: How many times to retry a failed generation call.
        generation_kwargs: Extra keyword arguments forwarded to
            ``backbone.generate()``.
        metadata: Arbitrary user-defined key-value pairs.
    """

    agent_id: str
    role: str = "child"
    domain: str = "general"
    task: str = "general"
    enabled_skills: list[str] = field(default_factory=list)
    adapter_type: str | None = None
    adapter_params: dict[str, Any] = field(default_factory=dict)
    max_retries: int = 1
    generation_kwargs: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def derive(self, **overrides: Any) -> "AgentConfig":
        """Return a shallow copy with the given fields overridden.

        This is the primary mechanism for creating child configurations
        from a mother-agent template without mutating the original.

        Example::

            base = AgentConfig(agent_id="mother", role="mother")
            child_cfg = base.derive(
                agent_id="child-medical",
                role="child",
                domain="medical",
                adapter_type="prompt",
            )
        """
        data = copy.deepcopy(self.__dict__)
        data.update(overrides)
        return AgentConfig(**data)
