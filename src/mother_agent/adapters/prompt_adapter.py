"""Prompt-template adapter — cheapest form of child-agent customisation."""

from __future__ import annotations

from string import Template
from typing import Any

from .base_adapter import AdapterConfig, BaseAdapter


class PromptAdapter(BaseAdapter):
    """Wraps prompts with a configurable system prefix and/or suffix.

    This is the lightest customisation possible: no additional parameters
    are trained; only the text that enters the backbone is modified.

    Configuration keys (``AdapterConfig.params``):

    * ``system_prefix`` (*str*): Text prepended to every prompt.
      Supports ``$domain`` and ``$task`` template variables.
    * ``system_suffix`` (*str*): Text appended to every prompt.
    * ``domain`` (*str*): Domain label substituted into template strings.
    * ``task`` (*str*): Task label substituted into template strings.
    * ``output_prefix`` (*str*): Optional prefix for post-processed output.
    * ``output_suffix`` (*str*): Optional suffix for post-processed output.

    Example::

        config = AdapterConfig(
            adapter_id="medical-qa",
            params={
                "system_prefix": "You are a medical assistant in the $domain domain.\\n",
                "domain": "cardiology",
                "task": "question_answering",
            },
        )
        adapter = PromptAdapter(config)
        adapted = adapter.adapt_prompt("What are the symptoms of angina?")
    """

    def adapt_prompt(self, prompt: str, context: dict[str, Any] | None = None) -> str:
        params = self.config.params
        prefix_template = params.get("system_prefix", "")
        suffix_template = params.get("system_suffix", "")

        substitutions = {
            "domain": params.get("domain", "general"),
            "task": params.get("task", "general"),
        }

        prefix = Template(prefix_template).safe_substitute(substitutions)
        suffix = Template(suffix_template).safe_substitute(substitutions)

        return f"{prefix}{prompt}{suffix}"

    def adapt_output(self, output: str, context: dict[str, Any] | None = None) -> str:
        params = self.config.params
        out_prefix = params.get("output_prefix", "")
        out_suffix = params.get("output_suffix", "")
        return f"{out_prefix}{output}{out_suffix}"
