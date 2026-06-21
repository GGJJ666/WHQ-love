"""Tests for MotherAgent and ChildAgent."""

from __future__ import annotations

import pytest

from mother_agent.backbone.base_backbone import BackboneOutput, MockBackbone
from mother_agent.child_agent import ChildAgent
from mother_agent.config import AgentConfig
from mother_agent.mother_agent import MotherAgent
from mother_agent.skills.base_skill import SkillResult
from mother_agent.skills.nlu_skill import NLUSkill


# ---------------------------------------------------------------------------
# MotherAgent tests
# ---------------------------------------------------------------------------


class TestMotherAgent:
    def setup_method(self):
        self.mother = MotherAgent()

    def test_default_agent_id(self):
        assert self.mother.config.agent_id == "mother"

    def test_default_skills_registered(self):
        skills = self.mother.available_skills
        assert "nlu" in skills
        assert "planning" in skills
        assert "tool_calling" in skills

    def test_register_custom_skill(self):
        class MySkill(NLUSkill):
            @property
            def name(self):
                return "my_custom_skill"

        self.mother.register_skill(MySkill())
        assert "my_custom_skill" in self.mother.available_skills

    def test_unregister_skill(self):
        self.mother.unregister_skill("nlu")
        assert "nlu" not in self.mother.available_skills

    def test_unregister_nonexistent_raises(self):
        with pytest.raises(KeyError):
            self.mother.unregister_skill("no_such_skill")

    def test_get_skill_returns_skill(self):
        skill = self.mother.get_skill("nlu")
        assert skill.name == "nlu"

    def test_get_skill_nonexistent_raises(self):
        with pytest.raises(KeyError):
            self.mother.get_skill("no_such_skill")

    def test_run_backbone(self):
        result = self.mother.run("Hello")
        assert isinstance(result, BackboneOutput)
        assert "Hello" in result.text

    def test_run_with_skill(self):
        result = self.mother.run("Search for cats", skill_name="nlu")
        assert isinstance(result, SkillResult)
        assert result.data["intent"] == "search"

    def test_spawn_child_registers_child(self):
        cfg = AgentConfig(agent_id="child-1", domain="finance")
        child = self.mother.spawn_child(cfg)
        assert "child-1" in self.mother.children

    def test_spawn_duplicate_raises(self):
        cfg = AgentConfig(agent_id="child-dup", domain="finance")
        self.mother.spawn_child(cfg)
        with pytest.raises(ValueError, match="already exists"):
            self.mother.spawn_child(cfg)

    def test_get_child_returns_child(self):
        cfg = AgentConfig(agent_id="child-get")
        self.mother.spawn_child(cfg)
        child = self.mother.get_child("child-get")
        assert child.config.agent_id == "child-get"

    def test_get_child_nonexistent_raises(self):
        with pytest.raises(KeyError):
            self.mother.get_child("ghost")

    def test_deregister_child(self):
        cfg = AgentConfig(agent_id="child-del")
        self.mother.spawn_child(cfg)
        self.mother.deregister_child("child-del")
        assert "child-del" not in self.mother.children

    def test_broadcast_skill_update(self):
        new_nlu = NLUSkill()
        self.mother.broadcast_skill_update(new_nlu)
        assert self.mother.get_skill("nlu") is new_nlu

    def test_repr(self):
        assert "MotherAgent" in repr(self.mother)

    def test_custom_backbone(self):
        bb = MockBackbone(model_name="custom-bb")
        mother = MotherAgent(backbone=bb)
        assert mother.backbone.model_name == "custom-bb"


# ---------------------------------------------------------------------------
# ChildAgent tests
# ---------------------------------------------------------------------------


class TestChildAgent:
    def setup_method(self):
        self.mother = MotherAgent()

    def _spawn(self, **kwargs) -> ChildAgent:
        defaults = {"agent_id": "test-child"}
        defaults.update(kwargs)
        return self.mother.spawn_child(AgentConfig(**defaults))

    def test_child_repr(self):
        child = self._spawn(agent_id="repr-child")
        assert "repr-child" in repr(child)

    def test_run_without_adapter(self):
        child = self._spawn(agent_id="plain-child")
        result = child.run("hello")
        assert isinstance(result, BackboneOutput)

    def test_run_with_prompt_adapter(self):
        child = self._spawn(
            agent_id="prompt-child",
            adapter_type="prompt",
            adapter_params={"system_prefix": "[MED] "},
        )
        result = child.run("What is fever?")
        assert isinstance(result, BackboneOutput)
        # The adapted prompt should appear in the backbone echo
        assert "[MED]" in result.text

    def test_run_with_lora_adapter(self):
        child = self._spawn(
            agent_id="lora-child",
            adapter_type="lora",
            adapter_params={"task_description": "Finance specialist"},
        )
        result = child.run("What is inflation?")
        assert isinstance(result, BackboneOutput)
        assert "lora-child" in result.text

    def test_invalid_adapter_type_raises(self):
        with pytest.raises(ValueError, match="Unknown adapter_type"):
            self._spawn(agent_id="bad-adapter", adapter_type="invalid")

    def test_run_with_skill(self):
        child = self._spawn(agent_id="skill-child")
        result = child.run("Plan a trip to Tokyo", skill_name="planning")
        assert isinstance(result, SkillResult)

    def test_skill_restriction(self):
        child = self._spawn(
            agent_id="restricted-child",
            enabled_skills=["planning"],
        )
        # planning is allowed
        result = child.run("Plan something", skill_name="planning")
        assert isinstance(result, SkillResult)

        # nlu is blocked
        with pytest.raises(KeyError, match="not enabled"):
            child.run("search query", skill_name="nlu")

    def test_available_skills_no_restriction(self):
        child = self._spawn(agent_id="all-skills-child")
        assert set(child.available_skills) == set(self.mother.available_skills)

    def test_available_skills_with_restriction(self):
        child = self._spawn(
            agent_id="restricted-skills",
            enabled_skills=["nlu"],
        )
        assert child.available_skills == ["nlu"]

    def test_checkpoint_roundtrip(self):
        child = self._spawn(
            agent_id="ckpt-child",
            adapter_type="lora",
            domain="medical",
            task="qa",
        )
        ckpt = child.get_checkpoint()
        assert ckpt["agent_id"] == "ckpt-child"
        assert ckpt["domain"] == "medical"
        assert "lora_state" in ckpt

    def test_child_shares_mother_skills(self):
        # Register a new skill on mother — child should see it
        class SpecialSkill(NLUSkill):
            @property
            def name(self):
                return "special"

        child = self._spawn(agent_id="sharing-child")
        self.mother.register_skill(SpecialSkill())
        assert "special" in child.available_skills

    def test_config_derive(self):
        base = AgentConfig(agent_id="base", domain="general")
        derived = base.derive(agent_id="derived", domain="medical")
        assert derived.agent_id == "derived"
        assert derived.domain == "medical"
        # unchanged field preserved
        assert derived.role == base.role
