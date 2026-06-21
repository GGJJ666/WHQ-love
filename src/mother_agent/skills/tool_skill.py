"""Tool-calling skill — routes requests to registered external tools."""

from __future__ import annotations

from typing import Any, Callable

from .base_skill import BaseSkill, SkillResult

# Type alias for a tool handler function.
ToolHandler = Callable[[str, dict[str, Any]], Any]


class ToolSkill(BaseSkill):
    r"""Maintains a registry of named tools and dispatches calls to them.

    Tools are plain Python callables registered via :meth:`register_tool`.
    Each callable receives ``(prompt, context)`` and returns any JSON-
    serialisable value.

    Example::

        tool_skill = ToolSkill()

        @tool_skill.register_tool("calculator")
        def calculator(prompt, context):
            # naive eval — do NOT use in production!
            import re
            expr = re.search(r"[\d+\-*/().\s]+", prompt)
            return eval(expr.group()) if expr else None

        result = tool_skill.execute("What is 2 + 3?", {"tool": "calculator"})
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolHandler] = {}

    @property
    def name(self) -> str:
        return "tool_calling"

    @property
    def description(self) -> str:
        return "Routes execution to registered external tools by name."

    def register_tool(self, tool_name: str) -> Callable[[ToolHandler], ToolHandler]:
        """Decorator that registers a callable as a named tool.

        Args:
            tool_name: Identifier used to look up the tool at runtime.

        Returns:
            The original function, unchanged.
        """

        def decorator(fn: ToolHandler) -> ToolHandler:
            self._tools[tool_name] = fn
            return fn

        return decorator

    def add_tool(self, tool_name: str, handler: ToolHandler) -> None:
        """Register *handler* under *tool_name* (non-decorator variant)."""
        self._tools[tool_name] = handler

    @property
    def available_tools(self) -> list[str]:
        """Names of all currently registered tools."""
        return list(self._tools.keys())

    def execute(self, prompt: str, context: dict[str, Any] | None = None) -> SkillResult:
        ctx = context or {}
        tool_name: str | None = ctx.get("tool")

        if tool_name is None:
            return SkillResult(
                skill_name=self.name,
                output=f"Available tools: {self.available_tools}",
                data={"available_tools": self.available_tools},
            )

        if tool_name not in self._tools:
            return SkillResult(
                skill_name=self.name,
                output=f"Tool '{tool_name}' not found.",
                data={"available_tools": self.available_tools},
                success=False,
                error=f"Unknown tool: {tool_name}",
            )

        try:
            result = self._tools[tool_name](prompt, ctx)
            return SkillResult(
                skill_name=self.name,
                output=str(result),
                data={"tool": tool_name, "result": result},
            )
        except Exception as exc:  # noqa: BLE001
            return SkillResult(
                skill_name=self.name,
                output=f"Tool '{tool_name}' raised an error: {exc}",
                data={"tool": tool_name},
                success=False,
                error=str(exc),
            )
