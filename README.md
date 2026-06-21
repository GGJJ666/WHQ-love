# WHQ-love

所有我们自己需要的技能池。  
*A skill pool for everything we need.*

---

## 母—子 AI Agent 架构 / Mother-Child AI Agent Architecture

本项目实现了一个**母—子多代理生成框架**（Mother-Child Multi-Agent Architecture），用于探索和验证以下核心假设：

> 一个"母 AI Agent"拥有共享骨干模型（backbone）与通用技能模块；通过轻量的适配层（Adapter），可快速生成多个高质量"子 AI Agent"，同时避免重复训练庞大的模型权重，显著降低资源消耗与维护复杂度。

### 架构概览

```
MotherAgent
├── backbone: BaseBackbone          ← 共享重量级模型（支持替换为真实 LLM）
├── skills: { nlu, planning, tool_calling, … }  ← 共享可插拔技能模块
└── children:
    ├── ChildAgent(domain="medical",  adapter=PromptAdapter)
    ├── ChildAgent(domain="finance",  adapter=LoRAAdapter)
    └── ChildAgent(domain="general",  adapter=None)
```

---

## 快速上手 / Quick Start

```bash
# 安装 / Install
pip install -e ".[dev]"

# 运行演示 / Run the demo
python examples/demo.py

# 运行测试 / Run tests
pytest
```

### 基本用法 / Basic Usage

```python
from mother_agent import MotherAgent
from mother_agent.config import AgentConfig

# 1. 创建母 Agent（默认使用 MockBackbone，可替换为真实模型）
mother = MotherAgent()

# 2. 派生子 Agent（Prompt 适配器）
child_medical = mother.spawn_child(AgentConfig(
    agent_id="child-medical",
    domain="medical",
    task="question_answering",
    adapter_type="prompt",
    adapter_params={"system_prefix": "You are a medical assistant.\n"},
))

# 3. 推理
result = child_medical.run("What are the symptoms of hypertension?")
print(result.text)

# 4. 技能调用（共享）
plan = child_medical.run("Diagnose, treat, then follow up", skill_name="planning")
print(plan.output)
```

---

## 项目结构 / Project Structure

```
src/mother_agent/
├── __init__.py              ← 公共 API
├── mother_agent.py          ← MotherAgent：拥有 backbone 与技能注册表
├── child_agent.py           ← ChildAgent：轻量派生，含适配器管道
├── config.py                ← AgentConfig：不可变配置数据类
├── evaluation.py            ← 评估工具：BenchmarkTask、Evaluator、干扰指标
├── backbone/
│   └── base_backbone.py     ← BaseBackbone 抽象类 + MockBackbone 存根
├── skills/
│   ├── base_skill.py        ← BaseSkill 抽象类
│   ├── nlu_skill.py         ← 意图识别与实体抽取
│   ├── planning_skill.py    ← 任务分解与步骤规划
│   └── tool_skill.py        ← 工具注册与调度
└── adapters/
    ├── base_adapter.py      ← BaseAdapter 抽象类
    ├── prompt_adapter.py    ← 系统提示模板适配器
    └── lora_adapter.py      ← LoRA 风格参数高效适配器（概念实现）

tests/
├── test_backbone.py
├── test_skills.py
├── test_adapters.py
├── test_agents.py
└── test_evaluation.py

examples/
└── demo.py                  ← 端到端演示脚本
```

---

## 核心组件说明 / Core Components

### MotherAgent

- 持有唯一的 `BaseBackbone` 实例（重量级共享模型）。
- 维护技能注册表（`register_skill` / `unregister_skill`）。
- 通过 `spawn_child(AgentConfig)` 生成并追踪子 Agent。
- 支持 `broadcast_skill_update` 将技能更新批量推送给所有子 Agent。

### ChildAgent

- 持有母 Agent 的**引用**，而非独立副本——不复制 backbone 权重。
- 通过 **PromptAdapter**（零参数开销）或 **LoRAAdapter**（小规模 delta 权重）定制行为。
- 可通过 `enabled_skills` 限制可用技能，实现权限隔离。
- 提供 `get_checkpoint()` 序列化轻量状态（仅适配器 + 配置）。

### 适配器 / Adapters

| 类型 | 描述 | 参数开销 |
|------|------|---------|
| `PromptAdapter` | 系统提示前缀/后缀，支持 `$domain`、`$task` 模板变量 | 零额外参数 |
| `LoRAAdapter` | 模拟 LoRA 低秩 delta 权重，支持检查点序列化/恢复 | `rank × 2` 个浮点数 |

### 评估工具 / Evaluation

```python
from mother_agent import Evaluator, BenchmarkTask, keyword_scorer

evaluator = Evaluator()
tasks = [
    BenchmarkTask(
        name="intent_detection",
        prompt="Search for AI papers",
        skill_name="nlu",
        scorer=keyword_scorer,
        scorer_kwargs={"keywords": ["search"]},
    )
]
report = evaluator.evaluate(agent, tasks)
print(report.summary())
# → Agent: mother | Tasks: 1 | Success rate: 100.0% | Mean latency: 0.0 ms | ...
```

**可量化指标**：
- `task_success_rate` — 任务成功率
- `mean_latency_ms` / `p95_latency_ms` — 响应延迟
- `mean_score` — 任务质量评分（可替换为任意评分函数）
- `cross_task_interference` — 跨任务干扰度
- `knowledge_isolation_score` — 知识隔离度

---

## 对接真实 LLM / Plugging in a Real LLM

只需继承 `BaseBackbone` 并传入 `MotherAgent`：

```python
from mother_agent.backbone.base_backbone import BaseBackbone, BackboneOutput
import openai

class OpenAIBackbone(BaseBackbone):
    def __init__(self, model: str = "gpt-4o-mini"):
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model

    def generate(self, prompt: str, **kwargs) -> BackboneOutput:
        response = openai.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        return BackboneOutput(text=response.choices[0].message.content)

    def embed(self, text: str) -> list[float]:
        resp = openai.embeddings.create(model="text-embedding-3-small", input=text)
        return resp.data[0].embedding

from mother_agent import MotherAgent
mother = MotherAgent(backbone=OpenAIBackbone())
```

---

## 可行性总结 / Feasibility Summary

| 维度 | 结论 |
|------|------|
| **架构可行性** | ✅ 共享 backbone + 适配器层可快速生成高质量子 Agent |
| **性能与效率** | ✅ 参数共享显著降低存储与训练成本；LoRA-style delta 极小 |
| **科学性** | ✅ 迁移学习、元学习、参数高效微调（LoRA/Adapter）均有坚实理论支撑 |
| **可扩展性** | ✅ 子 Agent 数量线性增长，边际资源成本接近零 |
| **风险** | ⚠️ 单点故障（母 Agent）；需沙箱、版本签名、权限隔离 |
| **建议** | 分阶段实验：小规模原型 → 多域对比 → 干扰隔离验证 |

---

## 运行测试 / Running Tests

```bash
pytest                     # 运行所有测试（94 个）
pytest tests/test_agents.py -v   # 仅运行 Agent 测试
```

## 许可证 / License

GNU General Public License v3.0 — 详见 [LICENSE](LICENSE)。
