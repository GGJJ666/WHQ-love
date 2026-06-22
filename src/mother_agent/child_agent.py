"""Child Agent — a lightweight specialisation spawned by a MotherAgent."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from .adapters.base_adapter import AdapterConfig, BaseAdapter
from .adapters.lora_adapter import LoRAAdapter
from .adapters.prompt_adapter import PromptAdapter
from .backbone.base_backbone import BackboneOutput
from .config import AgentConfig
from .skills.base_skill import SkillResult

if TYPE_CHECKING:
    from .mother_agent import MotherAgent


class ChildAgent:
    """A specialised agent that shares its mother's backbone and skills.

    A :class:`ChildAgent` is always owned by a :class:`~mother_agent.MotherAgent`
    and should be created via :meth:`~mother_agent.MotherAgent.spawn_child`
    rather than directly.

    Customisation is achieved through one of the following adapters (selected
    via :attr:`~config.AgentConfig.adapter_type`):

    * ``"prompt"`` — :class:`~adapters.PromptAdapter`: system-prompt prefix/suffix.
    * ``"lora"`` — :class:`~adapters.LoRAAdapter`: simulated parameter-efficient delta.
    * ``None`` — no adapter; requests pass through to the backbone unchanged.

    Args:
        config: Lightweight configuration describing this child's
            specialisation.
        mother: The :class:`~mother_agent.MotherAgent` that spawned this child.
    """

    def __init__(self, config: AgentConfig, mother: "MotherAgent") -> None:
        self.config = config
        self._mother = mother
        self._adapter: BaseAdapter | None = self._build_adapter()

    # ------------------------------------------------------------------
    # Adapter construction
    # ------------------------------------------------------------------

    def _build_adapter(self) -> BaseAdapter | None:
        adapter_type = self.config.adapter_type
        if adapter_type is None:
            return None

        adapter_cfg = AdapterConfig(
            adapter_id=self.config.agent_id,
            params={
                **self.config.adapter_params,
                "domain": self.config.domain,
                "task": self.config.task,
            },
        )
        if adapter_type == "prompt":
            return PromptAdapter(adapter_cfg)
        if adapter_type == "lora":
            return LoRAAdapter(adapter_cfg)
        raise ValueError(
            f"Unknown adapter_type '{adapter_type}'. "
            "Valid options: 'prompt', 'lora', None."
        )

    # ------------------------------------------------------------------
    # Skill access (delegated to mother's registry)
    # ------------------------------------------------------------------

    def get_skill(self, skill_name: str):
        """Retrieve a skill from the mother's shared registry.

        Args:
            skill_name: Skill identifier.

        Raises:
            KeyError: If the skill is not registered or not enabled for this
                child.
        """
        enabled = self.config.enabled_skills
        if enabled and skill_name not in enabled:
            raise KeyError(
                f"Skill '{skill_name}' is not enabled for child "
                f"'{self.config.agent_id}'. "
                f"Enabled skills: {enabled}"
            )
        return self._mother.get_skill(skill_name)

    @property
    def available_skills(self) -> list[str]:
        """Names of skills available to this child.

        If ``enabled_skills`` is empty the child can access all of the
        mother's registered skills.
        """
        all_skills = self._mother.available_skills
        if not self.config.enabled_skills:
            return all_skills
        return [s for s in self.config.enabled_skills if s in all_skills]

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def run(
        self,
        prompt: str,
        skill_name: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> BackboneOutput | SkillResult:
        """Execute a prompt through this child's customisation pipeline.

        Processing pipeline:

        1. If *skill_name* is provided, route to the corresponding skill.
        2. Apply the adapter's :meth:`~adapters.BaseAdapter.adapt_prompt`.
        3. Call the shared backbone's ``generate`` method.
        4. Apply the adapter's :meth:`~adapters.BaseAdapter.adapt_output`.

        Args:
            prompt: User input or sub-task description.
            skill_name: Optional skill to route the request to.
            context: Optional context dictionary.

        Returns:
            :class:`~backbone.BackboneOutput` or :class:`~skills.SkillResult`.
        """
        if skill_name is not None:
            skill = self.get_skill(skill_name)
            skill_context = {
                "agent_metadata": dict(self.config.metadata),
                **(context or {}),
            }
            return skill.execute(prompt, skill_context)

        adapted_prompt = (
            self._adapter.adapt_prompt(prompt, context)
            if self._adapter
            else prompt
        )

        start = time.perf_counter()
        output: BackboneOutput = self._mother.backbone.generate(
            adapted_prompt,
            **self.config.generation_kwargs,
        )
        output.latency_ms = (time.perf_counter() - start) * 1000

        if self._adapter:
            output.text = self._adapter.adapt_output(output.text, context)

        return output

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def get_checkpoint(self) -> dict[str, Any]:
        """Serialise the child's lightweight state for persistence.

        Only the *delta* (adapter + config) is serialised; the heavy backbone
        weights live on the mother and are not duplicated.

        Returns:
            JSON-serialisable dictionary.
        """
        checkpoint: dict[str, Any] = {
            "agent_id": self.config.agent_id,
            "role": self.config.role,
            "domain": self.config.domain,
            "task": self.config.task,
            "enabled_skills": self.config.enabled_skills,
            "adapter_type": self.config.adapter_type,
            "adapter_params": self.config.adapter_params,
            "generation_kwargs": self.config.generation_kwargs,
            "metadata": self.config.metadata,
        }
        if isinstance(self._adapter, LoRAAdapter):
            checkpoint["lora_state"] = self._adapter.get_checkpoint()
        return checkpoint

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"ChildAgent(id={self.config.agent_id!r}, "
            f"domain={self.config.domain!r}, "
            f"adapter={self.config.adapter_type!r})"
        )
