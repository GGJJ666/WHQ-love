"""Tests for the adapter modules."""

from __future__ import annotations

import pytest

from mother_agent.adapters.base_adapter import AdapterConfig
from mother_agent.adapters.lora_adapter import LoRAAdapter
from mother_agent.adapters.prompt_adapter import PromptAdapter


class TestPromptAdapter:
    def _make_adapter(self, **params) -> PromptAdapter:
        cfg = AdapterConfig(adapter_id="test-prompt", params=params)
        return PromptAdapter(cfg)

    def test_prefix_applied(self):
        adapter = self._make_adapter(system_prefix="[SYSTEM] ")
        result = adapter.adapt_prompt("hello")
        assert result.startswith("[SYSTEM] ")

    def test_suffix_applied(self):
        adapter = self._make_adapter(system_suffix=" [END]")
        result = adapter.adapt_prompt("hello")
        assert result.endswith(" [END]")

    def test_template_substitution(self):
        adapter = self._make_adapter(
            system_prefix="Domain: $domain. Task: $task.\n",
            domain="medical",
            task="qa",
        )
        result = adapter.adapt_prompt("What is blood pressure?")
        assert "medical" in result
        assert "qa" in result

    def test_no_prefix_no_suffix(self):
        adapter = self._make_adapter()
        result = adapter.adapt_prompt("original")
        assert result == "original"

    def test_output_prefix_suffix(self):
        adapter = self._make_adapter(
            output_prefix=">> ",
            output_suffix=" <<",
        )
        result = adapter.adapt_output("answer")
        assert result == ">> answer <<"

    def test_repr(self):
        adapter = self._make_adapter()
        assert "PromptAdapter" in repr(adapter)


class TestLoRAAdapter:
    def _make_adapter(self, **params) -> LoRAAdapter:
        cfg = AdapterConfig(adapter_id="test-lora", params=params)
        return LoRAAdapter(cfg)

    def test_default_rank(self):
        adapter = self._make_adapter()
        assert adapter.rank == 4

    def test_custom_rank(self):
        adapter = self._make_adapter(rank=8)
        assert adapter.rank == 8

    def test_delta_weights_length(self):
        adapter = self._make_adapter(rank=4)
        assert len(adapter.delta_weights) == 8  # rank * 2

    def test_adapt_prompt_includes_id(self):
        adapter = self._make_adapter()
        result = adapter.adapt_prompt("query")
        assert "test-lora" in result

    def test_adapt_prompt_with_task_description(self):
        adapter = self._make_adapter(task_description="Medical QA specialist")
        result = adapter.adapt_prompt("What is hypertension?")
        assert "Medical QA specialist" in result

    def test_adapt_embedding_changes_values(self):
        adapter = self._make_adapter()
        original = [0.5] * 128
        adapted = adapter.adapt_embedding(original)
        assert adapted != original

    def test_adapt_embedding_same_length(self):
        adapter = self._make_adapter()
        original = [0.5] * 128
        adapted = adapter.adapt_embedding(original)
        assert len(adapted) == len(original)

    def test_checkpoint_roundtrip(self):
        adapter = self._make_adapter(rank=4, alpha=2.0, task_description="test task")
        ckpt = adapter.get_checkpoint()
        restored = LoRAAdapter.from_checkpoint(ckpt)
        assert restored.config.adapter_id == adapter.config.adapter_id
        assert restored.rank == adapter.rank
        assert restored.alpha == adapter.alpha
        assert restored.delta_weights == adapter.delta_weights

    def test_custom_delta_weights(self):
        deltas = [0.01] * 8
        adapter = self._make_adapter(rank=4, delta_weights=deltas)
        assert adapter.delta_weights == deltas
