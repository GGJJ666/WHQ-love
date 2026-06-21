"""Tests for the evaluation module."""

from __future__ import annotations

import pytest

from mother_agent import MotherAgent
from mother_agent.config import AgentConfig
from mother_agent.evaluation import (
    BenchmarkReport,
    BenchmarkTask,
    Evaluator,
    TaskResult,
    cross_task_interference,
    exact_match_scorer,
    keyword_scorer,
    knowledge_isolation_score,
    substring_scorer,
)


# ---------------------------------------------------------------------------
# Scorer tests
# ---------------------------------------------------------------------------


class TestScorers:
    def test_exact_match_pass(self):
        assert exact_match_scorer("hello", "hello") == 1.0

    def test_exact_match_fail(self):
        assert exact_match_scorer("hello", "world") == 0.0

    def test_exact_match_strips_whitespace(self):
        assert exact_match_scorer("  hello  ", "hello") == 1.0

    def test_substring_pass(self):
        assert substring_scorer("The sky is blue", "sky") == 1.0

    def test_substring_fail(self):
        assert substring_scorer("The sky is blue", "ocean") == 0.0

    def test_substring_case_insensitive(self):
        assert substring_scorer("The Sky is Blue", "sky") == 1.0

    def test_keyword_all_present(self):
        score = keyword_scorer("The sky is blue and sunny", ["sky", "blue", "sunny"])
        assert score == 1.0

    def test_keyword_partial(self):
        score = keyword_scorer("The sky is blue", ["sky", "ocean"])
        assert score == 0.5

    def test_keyword_none_present(self):
        score = keyword_scorer("nothing relevant", ["sky", "ocean"])
        assert score == 0.0

    def test_keyword_empty_list(self):
        score = keyword_scorer("anything", [])
        assert score == 0.0


# ---------------------------------------------------------------------------
# Evaluator tests
# ---------------------------------------------------------------------------


class TestEvaluator:
    def setup_method(self):
        self.mother = MotherAgent()
        self.evaluator = Evaluator()

    def test_evaluate_returns_benchmark_report(self):
        tasks = [
            BenchmarkTask(
                name="task1",
                prompt="Search for cats",
                skill_name="nlu",
                scorer=keyword_scorer,
                scorer_kwargs={"keywords": ["search"]},
            )
        ]
        report = self.evaluator.evaluate(self.mother, tasks)
        assert isinstance(report, BenchmarkReport)
        assert report.total_tasks == 1

    def test_evaluate_success_rate(self):
        tasks = [
            BenchmarkTask(name=f"t{i}", prompt="hello")
            for i in range(4)
        ]
        report = self.evaluator.evaluate(self.mother, tasks)
        assert report.task_success_rate == 1.0

    def test_evaluate_latency_populated(self):
        tasks = [BenchmarkTask(name="latency_task", prompt="test")]
        report = self.evaluator.evaluate(self.mother, tasks)
        assert report.mean_latency_ms >= 0
        assert report.p95_latency_ms >= 0

    def test_compare_multiple_agents(self):
        child_cfg = AgentConfig(agent_id="eval-child", domain="test")
        child = self.mother.spawn_child(child_cfg)
        tasks = [BenchmarkTask(name="task", prompt="hello")]
        reports = self.evaluator.compare([self.mother, child], tasks)
        assert "mother" in reports
        assert "eval-child" in reports

    def test_failed_task_recorded(self):
        """Agents that raise exceptions should be recorded as failures."""

        class FailingAgent:
            class config:
                agent_id = "failing-agent"

            def run(self, prompt, skill_name=None, context=None):
                raise RuntimeError("deliberate failure")

        tasks = [BenchmarkTask(name="fail_task", prompt="trigger")]
        report = self.evaluator.evaluate(FailingAgent(), tasks)
        assert report.task_success_rate == 0.0

    def test_benchmark_report_summary_string(self):
        tasks = [BenchmarkTask(name="t", prompt="x")]
        report = self.evaluator.evaluate(self.mother, tasks)
        summary = report.summary()
        assert "mother" in summary
        assert "%" in summary


# ---------------------------------------------------------------------------
# Interference and isolation metric tests
# ---------------------------------------------------------------------------


class TestInterferenceMetrics:
    def test_cross_task_interference_no_degradation(self):
        baseline = {"task_a": 0.9, "task_b": 0.8}
        mixed = {"task_a": 0.9, "task_b": 0.8}
        result = cross_task_interference(baseline, mixed)
        assert all(abs(v) < 1e-6 for v in result.values())

    def test_cross_task_interference_degraded(self):
        baseline = {"task_a": 0.9}
        mixed = {"task_a": 0.6}
        result = cross_task_interference(baseline, mixed)
        assert result["task_a"] > 0

    def test_cross_task_interference_only_common_tasks(self):
        baseline = {"task_a": 0.9, "task_b": 0.8}
        mixed = {"task_a": 0.7}
        result = cross_task_interference(baseline, mixed)
        assert "task_a" in result
        assert "task_b" not in result

    def test_knowledge_isolation_perfect(self):
        # Identical scores → high correlation → lower isolation
        # Opposite scores → no correlation → higher isolation
        a = {"t1": 0.9, "t2": 0.1, "t3": 0.8, "t4": 0.2}
        b = {"t1": 0.1, "t2": 0.9, "t3": 0.2, "t4": 0.8}
        score = knowledge_isolation_score(a, b)
        assert score >= 0.5  # weakly isolated

    def test_knowledge_isolation_returns_in_range(self):
        a = {"t1": 0.5, "t2": 0.6}
        b = {"t1": 0.4, "t2": 0.7}
        score = knowledge_isolation_score(a, b)
        assert 0.0 <= score <= 1.0

    def test_knowledge_isolation_too_few_tasks(self):
        a = {"t1": 0.5}
        b = {"t1": 0.5}
        score = knowledge_isolation_score(a, b)
        assert score == 1.0  # single common task → undefined correlation → 1.0
