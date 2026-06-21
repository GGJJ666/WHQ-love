"""Tests for the backbone module."""

from __future__ import annotations

import pytest

from mother_agent.backbone.base_backbone import BackboneOutput, MockBackbone


class TestMockBackbone:
    def setup_method(self):
        self.backbone = MockBackbone()

    def test_model_name_default(self):
        assert self.backbone.model_name == "mock-backbone-v1"

    def test_custom_model_name(self):
        bb = MockBackbone(model_name="custom-model")
        assert bb.model_name == "custom-model"

    def test_generate_returns_backbone_output(self):
        result = self.backbone.generate("Hello, world!")
        assert isinstance(result, BackboneOutput)
        assert "Hello, world!" in result.text

    def test_generate_includes_latency(self):
        result = self.backbone.generate("test")
        assert result.latency_ms >= 0

    def test_generate_includes_embeddings(self):
        result = self.backbone.generate("test")
        assert result.embeddings is not None
        assert len(result.embeddings) == 128

    def test_embed_returns_128_dims(self):
        embedding = self.backbone.embed("hello")
        assert len(embedding) == 128

    def test_embed_deterministic(self):
        e1 = self.backbone.embed("same text")
        e2 = self.backbone.embed("same text")
        assert e1 == e2

    def test_embed_different_inputs_differ(self):
        e1 = self.backbone.embed("text one")
        e2 = self.backbone.embed("text two")
        assert e1 != e2

    def test_generate_passes_kwargs_to_metadata(self):
        result = self.backbone.generate("test", temperature=0.7)
        assert result.metadata.get("kwargs", {}).get("temperature") == 0.7
