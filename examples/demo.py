#!/usr/bin/env python3
"""End-to-end demonstration of the Mother-Child AI Agent architecture.

Run with:
    python examples/demo.py
"""

from __future__ import annotations

from mother_agent import (
    Evaluator,
    BenchmarkTask,
    MotherAgent,
    keyword_scorer,
    cross_task_interference,
)
from mother_agent.config import AgentConfig
from mother_agent.skills.tool_skill import ToolSkill


def main() -> None:
    print("=" * 60)
    print("母—子 AI Agent 架构演示 / Mother-Child AI Agent Demo")
    print("=" * 60)

    # ---------------------------------------------------------------
    # 1. Create the Mother Agent
    # ---------------------------------------------------------------
    print("\n[1] Creating Mother Agent …")
    mother = MotherAgent()
    print(f"    {mother}")

    # Register a custom tool on the shared ToolSkill
    tool_skill: ToolSkill = mother.get_skill("tool_calling")  # type: ignore[assignment]

    @tool_skill.register_tool("word_count")
    def word_count(prompt: str, ctx: dict) -> int:
        return len(prompt.split())

    print(f"    Available skills: {mother.available_skills}")

    # ---------------------------------------------------------------
    # 2. Spawn Child Agents with different specialisations
    # ---------------------------------------------------------------
    print("\n[2] Spawning child agents …")

    child_science = mother.spawn_child(
        AgentConfig(
            agent_id="child-science",
            domain="science",
            task="question_answering",
            adapter_type="prompt",
            adapter_params={
                "system_prefix": "[科学助手 / Science Assistant] Domain: $domain\n",
            },
        )
    )

    child_finance = mother.spawn_child(
        AgentConfig(
            agent_id="child-finance",
            domain="finance",
            task="analysis",
            adapter_type="lora",
            adapter_params={
                "rank": 8,
                "alpha": 2.0,
                "task_description": "金融分析专家 / Finance Analysis Specialist",
            },
        )
    )

    child_planning = mother.spawn_child(
        AgentConfig(
            agent_id="child-planner",
            domain="general",
            enabled_skills=["planning"],  # restricted to planning only
        )
    )

    print(f"    {child_science}")
    print(f"    {child_finance}")
    print(f"    {child_planning}")
    print(f"    Mother's children: {list(mother.children.keys())}")

    # ---------------------------------------------------------------
    # 3. Run inference
    # ---------------------------------------------------------------
    print("\n[3] Running inference …")

    q_science = "How does photosynthesis work in plants?"
    q_finance = "Explain the impact of interest rate hikes on bond prices."
    q_plan = "Buy groceries, then cook dinner, then watch a movie"

    print(f"\n  Science child — Q: {q_science!r}")
    r = child_science.run(q_science)
    print(f"  → {r.text}")

    print(f"\n  Finance child — Q: {q_finance!r}")
    r = child_finance.run(q_finance)
    print(f"  → {r.text}")

    print(f"\n  Planner child — Q: {q_plan!r}")
    r = child_planning.run(q_plan, skill_name="planning")
    print(f"  → {r.output}")

    # Shared tool available to all children
    print(f"\n  Word count tool — shared with all children")
    r_tool = child_science.run(
        q_science, skill_name="tool_calling", context={"tool": "word_count"}
    )
    print(f"  → word count = {r_tool.output}")

    # ---------------------------------------------------------------
    # 4. Benchmark evaluation
    # ---------------------------------------------------------------
    print("\n[4] Benchmarking agents …")
    evaluator = Evaluator()

    tasks = [
        BenchmarkTask(
            name="nlu_search",
            prompt="Search for machine learning papers",
            skill_name="nlu",
            scorer=keyword_scorer,
            scorer_kwargs={"keywords": ["search"]},
        ),
        BenchmarkTask(
            name="planning_trip",
            prompt="Book a flight, then reserve a hotel, then plan activities",
            skill_name="planning",
            scorer=keyword_scorer,
            scorer_kwargs={"keywords": ["Book", "reserve", "plan"]},
        ),
        BenchmarkTask(
            name="backbone_response",
            prompt="Describe the role of central banks",
        ),
    ]

    # Benchmark all agents (mother + children that support the tasks)
    reports = evaluator.compare([mother, child_science, child_finance], tasks)
    print()
    for agent_id, report in reports.items():
        print(f"  {report.summary()}")

    # ---------------------------------------------------------------
    # 5. Cross-task interference demo
    # ---------------------------------------------------------------
    print("\n[5] Measuring cross-task interference …")
    baseline_scores = {t.name: 0.8 for t in tasks}
    mixed_scores = {t.name: r.score for r, t in
                    zip(reports["mother"].results, tasks)}

    interference = cross_task_interference(baseline_scores, mixed_scores)
    print("  Task interference ratios (positive = degraded vs baseline 0.8):")
    for task_name, ratio in interference.items():
        print(f"    {task_name}: {ratio:+.3f}")

    # ---------------------------------------------------------------
    # 6. Child checkpoint
    # ---------------------------------------------------------------
    print("\n[6] Checkpointing child-finance …")
    ckpt = child_finance.get_checkpoint()
    print(f"  Checkpoint keys: {list(ckpt.keys())}")
    print(f"  LoRA rank: {ckpt['lora_state']['rank']}, "
          f"alpha: {ckpt['lora_state']['alpha']}")

    print("\n✓ Demo completed successfully.")


if __name__ == "__main__":
    main()
