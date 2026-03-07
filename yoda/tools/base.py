"""Abstract base class for agent tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from yoda.types import ToolResult


class ToolParameter(BaseModel):
    """Schema for a single tool parameter (used for LLM function-calling)."""

    name: str
    description: str
    type: str = "string"  # JSON Schema type
    required: bool = True
    default: Any = None


class ToolSpec(BaseModel):
    """Machine-readable tool specification for LLM function-calling."""

    name: str
    description: str
    parameters: list[ToolParameter] = Field(default_factory=list)

    def to_openai_schema(self) -> dict[str, Any]:
        """Convert to OpenAI-compatible function schema."""
        properties: dict[str, Any] = {}
        required: list[str] = []
        for param in self.parameters:
            properties[param.name] = {
                "type": param.type,
                "description": param.description,
            }
            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }


class BaseTool(ABC):
    """Async interface for an agent tool.

    Downstream agents should subclass this to implement concrete tools
    (filesystem, web search, etc.).
    """

    @property
    @abstractmethod
    def spec(self) -> ToolSpec:
        """Return the tool's specification for LLM function-calling."""

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def description(self) -> str:
        return self.spec.description

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Run the tool with the given arguments and return a result.

        Implementations should catch their own exceptions and return
        ToolResult(success=False, error=...) rather than raising.
        """

    def to_openai_schema(self) -> dict[str, Any]:
        """Convenience: get the OpenAI function-calling schema."""
        return self.spec.to_openai_schema()
