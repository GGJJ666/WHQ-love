"""Simulated LoRA-style adapter for parameter-efficient fine-tuning demonstration.

This module provides a *conceptual* implementation of a LoRA (Low-Rank
Adaptation) adapter.  In a real deployment this would inject trainable
low-rank weight matrices into transformer attention layers.  Here we
simulate the effect by maintaining a small dictionary of delta-parameters
that scale the backbone's output logits or embeddings.

The implementation is intentionally kept free of heavy ML-framework
dependencies so it can run in any Python environment.
"""

from __future__ import annotations

import math
from typing import Any

from .base_adapter import AdapterConfig, BaseAdapter


class LoRAAdapter(BaseAdapter):
    """Parameter-efficient adapter inspired by LoRA (Hu et al., 2021).

    A LoRA adapter stores two low-rank matrices **A** (d × r) and **B**
    (r × d) whose product **BA** approximates the weight delta ΔW.  The
    effective rank *r* is much smaller than the hidden dimension *d*,
    making this approach memory- and compute-efficient.

    This class simulates that behaviour by:

    1. Storing a small set of learnable "delta" values (``params`` dict).
    2. Applying a scaling function to a mock embedding vector.
    3. Providing serialisation / deserialisation for weight checkpointing.

    Configuration keys (``AdapterConfig.params``):

    * ``rank`` (*int*, default ``4``): LoRA rank *r*.
    * ``alpha`` (*float*, default ``1.0``): Scaling factor (``alpha / rank``).
    * ``delta_weights`` (*list[float]*): Pre-initialised delta vector
      (length == ``rank * 2``).  Random if not provided.
    * ``task_description`` (*str*): Used to prepend task context to prompts.
    """

    def __init__(self, config: AdapterConfig) -> None:
        super().__init__(config)
        params = config.params
        self.rank: int = params.get("rank", 4)
        self.alpha: float = params.get("alpha", 1.0)
        self.scaling: float = self.alpha / self.rank

        # Initialise delta weights deterministically from adapter_id seed
        seed = sum(ord(c) for c in config.adapter_id)
        if "delta_weights" in params:
            self._delta: list[float] = list(params["delta_weights"])
        else:
            self._delta = [
                math.sin(seed * (i + 1) * 0.1) * 0.01
                for i in range(self.rank * 2)
            ]

    @property
    def delta_weights(self) -> list[float]:
        """Current delta weight vector (read-only snapshot)."""
        return list(self._delta)

    def adapt_prompt(self, prompt: str, context: dict[str, Any] | None = None) -> str:
        task_desc = self.config.params.get("task_description", "")
        if task_desc:
            return f"[LoRA:{self.config.adapter_id}] {task_desc}\n{prompt}"
        return f"[LoRA:{self.config.adapter_id}] {prompt}"

    def adapt_embedding(self, embedding: list[float]) -> list[float]:
        """Apply a simulated LoRA delta to an embedding vector.

        In a real implementation this would perform the low-rank matrix
        multiplication.  Here we add a scaled perturbation derived from
        the stored delta weights.

        Args:
            embedding: Original embedding vector from the backbone.

        Returns:
            Perturbed embedding of the same length.
        """
        result = list(embedding)
        for i, delta in enumerate(self._delta):
            idx = i % len(result)
            result[idx] += delta * self.scaling
        return result

    def get_checkpoint(self) -> dict[str, Any]:
        """Serialise adapter state for checkpointing.

        Returns:
            Dictionary suitable for ``json.dumps``.
        """
        return {
            "adapter_id": self.config.adapter_id,
            "rank": self.rank,
            "alpha": self.alpha,
            "delta_weights": self._delta,
            "params": {
                k: v
                for k, v in self.config.params.items()
                if k != "delta_weights"
            },
        }

    @classmethod
    def from_checkpoint(cls, checkpoint: dict[str, Any]) -> "LoRAAdapter":
        """Restore a :class:`LoRAAdapter` from a serialised checkpoint.

        Args:
            checkpoint: Dictionary as returned by :meth:`get_checkpoint`.

        Returns:
            Reconstructed :class:`LoRAAdapter` instance.
        """
        config = AdapterConfig(
            adapter_id=checkpoint["adapter_id"],
            params={
                **checkpoint.get("params", {}),
                "rank": checkpoint["rank"],
                "alpha": checkpoint["alpha"],
                "delta_weights": checkpoint["delta_weights"],
            },
        )
        return cls(config)
