"""Tests for the skill modules."""

from __future__ import annotations

import pytest

from mother_agent.skills.base_skill import SkillResult
from mother_agent.skills.nlu_skill import NLUSkill
from mother_agent.skills.planning_skill import PlanningSkill
from mother_agent.skills.tool_skill import ToolSkill


class TestNLUSkill:
    def setup_method(self):
        self.skill = NLUSkill()

    def test_name(self):
        assert self.skill.name == "nlu"

    def test_returns_skill_result(self):
        result = self.skill.execute("Search for Python tutorials")
        assert isinstance(result, SkillResult)
        assert result.success is True

    def test_detects_search_intent(self):
        result = self.skill.execute("Search for Python tutorials")
        assert result.data["intent"] == "search"

    def test_detects_plan_intent(self):
        result = self.skill.execute("Plan a trip to Tokyo")
        assert result.data["intent"] == "plan"

    def test_unknown_intent_fallback(self):
        result = self.skill.execute("xyzzy frobble quux")
        assert result.data["intent"] == "unknown"

    def test_entity_extraction(self):
        result = self.skill.execute("Search for Paris weather")
        assert "Paris" in result.data["entities"]

    def test_extracts_product_specific_terms_from_context(self):
        result = self.skill.execute(
            "create a report for whq love and lovegraph",
            context={"product_terms": ["WHQ Love", "LoveGraph"]},
        )
        assert result.data["entities"] == ["WHQ Love", "LoveGraph"]

    def test_product_terms_can_match_later_non_overlapping_occurrence(self):
        result = self.skill.execute(
            "compare whq love with love",
            context={"product_terms": ["Love", "WHQ Love"]},
        )
        assert result.data["entities"] == ["WHQ Love", "Love"]

    def test_repr(self):
        assert "NLUSkill" in repr(self.skill)


class TestPlanningSkill:
    def setup_method(self):
        self.skill = PlanningSkill()

    def test_name(self):
        assert self.skill.name == "planning"

    def test_returns_skill_result(self):
        result = self.skill.execute("Buy milk, then go to the gym")
        assert isinstance(result, SkillResult)
        assert result.success is True

    def test_steps_in_data(self):
        result = self.skill.execute("Buy milk, then go to the gym")
        assert "steps" in result.data
        assert result.data["num_steps"] >= 1

    def test_max_steps_respected(self):
        result = self.skill.execute(
            "step1, step2, step3, step4, step5, step6",
            context={"max_steps": 3},
        )
        assert result.data["num_steps"] <= 3

    def test_fallback_plan_for_simple_prompt(self):
        result = self.skill.execute("do something")
        assert result.data["num_steps"] >= 1


class TestToolSkill:
    def setup_method(self):
        self.skill = ToolSkill()
        self.skill.add_tool("echo", lambda prompt, ctx: f"ECHO: {prompt}")

    def test_name(self):
        assert self.skill.name == "tool_calling"

    def test_list_tools_when_no_tool_in_context(self):
        result = self.skill.execute("anything")
        assert result.success is True
        assert "echo" in result.data["available_tools"]

    def test_dispatch_to_registered_tool(self):
        result = self.skill.execute("hello", context={"tool": "echo"})
        assert result.success is True
        assert "ECHO: hello" in result.output

    def test_unknown_tool_returns_error(self):
        result = self.skill.execute("hi", context={"tool": "nonexistent"})
        assert result.success is False

    def test_decorator_registration(self):
        @self.skill.register_tool("upper")
        def upper_tool(prompt, ctx):
            return prompt.upper()

        result = self.skill.execute("hello", context={"tool": "upper"})
        assert result.success is True
        assert "HELLO" in result.output

    def test_tool_exception_caught(self):
        self.skill.add_tool("broken", lambda p, c: 1 / 0)
        result = self.skill.execute("x", context={"tool": "broken"})
        assert result.success is False
        assert "ZeroDivisionError" in result.error or "division by zero" in result.error

    def test_available_tools_property(self):
        assert "echo" in self.skill.available_tools
