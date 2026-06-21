"""母—子 AI Agent 架构 (Mother-Child AI Agent Architecture)

A Python framework for building and evaluating hierarchical multi-agent
systems in which a central *Mother Agent* owns the shared model backbone and
spawns lightweight *Child Agents* that customise behaviour through adapters.

Quick-start
-----------

.. code-block:: python

    from mother_agent import MotherAgent
    from mother_agent.config import AgentConfig

    # 1. Create the mother agent (uses a mock backbone by default)
    mother = MotherAgent()

    # 2. Spawn a child agent with a prompt adapter
    child_config = AgentConfig(
        agent_id="child-medical",
        domain="medical",
        task="question_answering",
        adapter_type="prompt",
        adapter_params={
            "system_prefix": "You are a medical assistant.\n",
        },
    )
    child = mother.spawn_child(child_config)

    # 3. Run inference
    result = child.run("What are the symptoms of hypertension?")
    print(result.text)
"""

from .backbone import BaseBackbone, BackboneOutput, MockBackbone
from .child_agent import ChildAgent
from .config import AgentConfig
from .evaluation import (
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
from .mother_agent import MotherAgent
from .skills import BaseSkill, NLUSkill, PlanningSkill, SkillResult, ToolSkill

__all__ = [
    # Core agents
    "MotherAgent",
    "ChildAgent",
    # Configuration
    "AgentConfig",
    # Backbone
    "BaseBackbone",
    "BackboneOutput",
    "MockBackbone",
    # Skills
    "BaseSkill",
    "SkillResult",
    "NLUSkill",
    "PlanningSkill",
    "ToolSkill",
    # Evaluation
    "Evaluator",
    "BenchmarkTask",
    "BenchmarkReport",
    "TaskResult",
    "exact_match_scorer",
    "substring_scorer",
    "keyword_scorer",
    "cross_task_interference",
    "knowledge_isolation_score",
]
