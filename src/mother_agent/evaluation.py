"""Evaluation utilities — metrics and benchmarking for the Mother-Child architecture."""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .backbone.base_backbone import BackboneOutput
from .skills.base_skill import SkillResult


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class TaskResult:
    """Outcome of a single task evaluation run.

    Attributes:
        agent_id: ID of the agent that produced this result.
        task_name: Identifier for the evaluated task.
        success: Whether the task was completed successfully.
        latency_ms: Wall-clock time from prompt submission to final output.
        output: Text output produced by the agent.
        score: Optional numeric quality score in [0, 1].
        metadata: Arbitrary additional data for the task.
    """

    agent_id: str
    task_name: str
    success: bool
    latency_ms: float
    output: str
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkReport:
    """Aggregated report over a set of task evaluations.

    Attributes:
        results: Individual task results.
        agent_id: ID of the evaluated agent.
        task_success_rate: Fraction of tasks that succeeded.
        mean_latency_ms: Mean latency across all tasks.
        p95_latency_ms: 95th-percentile latency.
        mean_score: Mean quality score across all tasks.
        total_tasks: Total number of tasks evaluated.
    """

    results: list[TaskResult]
    agent_id: str
    task_success_rate: float = 0.0
    mean_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    mean_score: float = 0.0
    total_tasks: int = 0

    def __post_init__(self) -> None:
        if self.results:
            self._compute_aggregates()

    def _compute_aggregates(self) -> None:
        latencies = [r.latency_ms for r in self.results]
        scores = [r.score for r in self.results]
        successes = [r.success for r in self.results]

        self.total_tasks = len(self.results)
        self.task_success_rate = sum(successes) / self.total_tasks
        self.mean_latency_ms = statistics.mean(latencies)
        self.p95_latency_ms = _percentile(latencies, 95)
        self.mean_score = statistics.mean(scores) if scores else 0.0

    def summary(self) -> str:
        return (
            f"Agent: {self.agent_id} | "
            f"Tasks: {self.total_tasks} | "
            f"Success rate: {self.task_success_rate:.1%} | "
            f"Mean latency: {self.mean_latency_ms:.1f} ms | "
            f"P95 latency: {self.p95_latency_ms:.1f} ms | "
            f"Mean score: {self.mean_score:.3f}"
        )


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def exact_match_scorer(output: str, reference: str) -> float:
    """Return 1.0 if *output* exactly equals *reference*, else 0.0."""
    return 1.0 if output.strip() == reference.strip() else 0.0


def substring_scorer(output: str, reference: str) -> float:
    """Return 1.0 if *reference* is a substring of *output*, else 0.0."""
    return 1.0 if reference.strip().lower() in output.lower() else 0.0


def keyword_scorer(output: str, keywords: list[str]) -> float:
    """Return fraction of *keywords* found in *output* (case-insensitive)."""
    if not keywords:
        return 0.0
    lower_out = output.lower()
    hits = sum(1 for kw in keywords if kw.lower() in lower_out)
    return hits / len(keywords)


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkTask:
    """A single task definition used in benchmark evaluation.

    Attributes:
        name: Human-readable task name.
        prompt: Input prompt to send to the agent.
        skill_name: Optional skill to route the task to.
        context: Optional context dictionary.
        scorer: Callable that accepts ``(output, **kwargs)`` and returns a
            score in ``[0, 1]``.  Defaults to always returning 0.5.
        scorer_kwargs: Extra keyword arguments passed to *scorer*.
        expected_success: Whether the task is expected to succeed.
    """

    name: str
    prompt: str
    skill_name: str | None = None
    context: dict[str, Any] | None = None
    scorer: Callable[..., float] = field(default_factory=lambda: lambda out, **kw: 0.5)
    scorer_kwargs: dict[str, Any] = field(default_factory=dict)
    expected_success: bool = True


class Evaluator:
    """Runs a benchmark task suite against one or more agents.

    Example::

        evaluator = Evaluator()
        tasks = [
            BenchmarkTask(
                name="nlu_intent",
                prompt="Search for Python tutorials",
                skill_name="nlu",
                scorer=keyword_scorer,
                scorer_kwargs={"keywords": ["search"]},
            ),
        ]
        report = evaluator.evaluate(agent, tasks)
        print(report.summary())
    """

    def evaluate(
        self,
        agent: Any,
        tasks: list[BenchmarkTask],
    ) -> BenchmarkReport:
        """Run all *tasks* against *agent* and return an aggregated report.

        Args:
            agent: Any agent with a ``run(prompt, skill_name, context)``
                method and a ``config.agent_id`` attribute.
            tasks: List of :class:`BenchmarkTask` instances.

        Returns:
            :class:`BenchmarkReport` with per-task and aggregated metrics.
        """
        results: list[TaskResult] = []
        for task in tasks:
            result = self._run_task(agent, task)
            results.append(result)
        return BenchmarkReport(results=results, agent_id=agent.config.agent_id)

    def compare(
        self,
        agents: list[Any],
        tasks: list[BenchmarkTask],
    ) -> dict[str, BenchmarkReport]:
        """Run all *tasks* against multiple *agents* and compare results.

        Args:
            agents: List of agents to benchmark.
            tasks: Task suite.

        Returns:
            Dictionary mapping agent ID to its :class:`BenchmarkReport`.
        """
        return {agent.config.agent_id: self.evaluate(agent, tasks) for agent in agents}

    @staticmethod
    def _run_task(agent: Any, task: BenchmarkTask) -> TaskResult:
        start = time.perf_counter()
        try:
            raw = agent.run(
                prompt=task.prompt,
                skill_name=task.skill_name,
                context=task.context,
            )
            latency_ms = (time.perf_counter() - start) * 1000

            if isinstance(raw, BackboneOutput):
                output_text = raw.text
                success = True
            elif isinstance(raw, SkillResult):
                output_text = raw.output
                success = raw.success
            else:
                output_text = str(raw)
                success = True

            score = task.scorer(output_text, **task.scorer_kwargs)
        except Exception as exc:  # noqa: BLE001
            latency_ms = (time.perf_counter() - start) * 1000
            output_text = str(exc)
            success = False
            score = 0.0

        return TaskResult(
            agent_id=agent.config.agent_id,
            task_name=task.name,
            success=success,
            latency_ms=latency_ms,
            output=output_text,
            score=score,
        )


# ---------------------------------------------------------------------------
# Interference & isolation metrics
# ---------------------------------------------------------------------------


def cross_task_interference(
    baseline_scores: dict[str, float],
    mixed_scores: dict[str, float],
) -> dict[str, float]:
    """Measure per-task performance degradation after mixing tasks.

    Args:
        baseline_scores: Task-name → score when tasks run in isolation.
        mixed_scores: Task-name → score when tasks share an agent.

    Returns:
        Dictionary of task-name → interference ratio (positive = degraded).
    """
    result: dict[str, float] = {}
    for task in baseline_scores:
        if task in mixed_scores:
            baseline = baseline_scores[task]
            mixed = mixed_scores[task]
            result[task] = (baseline - mixed) / (baseline + 1e-9)
    return result


def knowledge_isolation_score(
    child_a_scores: dict[str, float],
    child_b_scores: dict[str, float],
) -> float:
    """Estimate how well two children remain isolated from each other.

    A score of 1.0 indicates perfect isolation (no cross-contamination);
    a score of 0.0 indicates complete mixing.

    The heuristic is 1 − (mean absolute cross-score), where cross-score is
    the correlation between each child's per-task deltas.

    Args:
        child_a_scores: Task-name → score for child A.
        child_b_scores: Task-name → score for child B.

    Returns:
        Isolation score in [0, 1].
    """
    common = sorted(set(child_a_scores) & set(child_b_scores))
    if len(common) < 2:
        return 1.0

    a_vals = [child_a_scores[t] for t in common]
    b_vals = [child_b_scores[t] for t in common]

    mean_a = statistics.mean(a_vals)
    mean_b = statistics.mean(b_vals)

    cov = statistics.mean((a - mean_a) * (b - mean_b) for a, b in zip(a_vals, b_vals))
    std_a = statistics.pstdev(a_vals) or 1e-9
    std_b = statistics.pstdev(b_vals) or 1e-9
    correlation = cov / (std_a * std_b)

    # Map correlation ∈ [-1, 1] to isolation ∈ [0, 1]
    return (1.0 - abs(correlation)) / 2 + 0.5


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _percentile(data: list[float], p: float) -> float:
    """Return the *p*-th percentile of *data* (0 ≤ p ≤ 100)."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = (p / 100) * (len(sorted_data) - 1)
    lo = int(idx)
    hi = lo + 1
    if hi >= len(sorted_data):
        return sorted_data[lo]
    frac = idx - lo
    return sorted_data[lo] * (1 - frac) + sorted_data[hi] * frac
