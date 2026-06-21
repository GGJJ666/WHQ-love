"""Base backbone interface for the shared model component."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class BackboneOutput:
    """Output produced by the backbone model.

    Attributes:
        text: Generated text response.
        embeddings: Optional dense vector representation of the input.
        metadata: Arbitrary metadata returned by the backbone.
        latency_ms: Wall-clock inference time in milliseconds.
    """

    text: str
    embeddings: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0


class BaseBackbone(ABC):
    """Abstract base class for the shared backbone model.

    Concrete implementations should wrap a real language model (e.g., via an
    OpenAI-compatible API, a local HuggingFace model, or any other inference
    engine).  The stub implementation :class:`MockBackbone` below can be used
    for unit tests and offline experimentation.
    """

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> BackboneOutput:
        """Generate a response for *prompt*.

        Args:
            prompt: Input text sent to the model.
            **kwargs: Backend-specific generation parameters (temperature,
                max_tokens, …).

        Returns:
            :class:`BackboneOutput` containing the generated text and optional
            metadata.
        """

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Return a dense embedding vector for *text*.

        Args:
            text: Input string to embed.

        Returns:
            List of floats representing the embedding.
        """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable name / identifier for this backbone."""


class MockBackbone(BaseBackbone):
    """Deterministic stub backbone used for testing and prototyping.

    This implementation does **not** call any external service.  It echoes a
    formatted version of the prompt so that downstream components can be
    exercised without a real language model.
    """

    def __init__(self, model_name: str = "mock-backbone-v1") -> None:
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(self, prompt: str, **kwargs: Any) -> BackboneOutput:
        start = time.perf_counter()
        response = f"[{self._model_name}] Response to: {prompt[:80]}"
        latency_ms = (time.perf_counter() - start) * 1000
        return BackboneOutput(
            text=response,
            embeddings=self.embed(prompt),
            metadata={"model": self._model_name, "kwargs": kwargs},
            latency_ms=latency_ms,
        )

    def embed(self, text: str) -> list[float]:
        """Return a simple hash-based mock embedding (128 dims)."""
        seed = sum(ord(c) for c in text)
        return [(seed * (i + 1) % 997) / 997.0 for i in range(128)]
