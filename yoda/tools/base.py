"""Base tool interface and data types for the Yoda tool framework."""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


class ParameterType(str, enum.Enum):
    """Supported parameter types for tool inputs."""

    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    LIST = "list"
    OBJECT = "object"


@dataclass(frozen=True, slots=True)
class ToolParameter:
    """Describes a single parameter accepted by a tool.

    Attributes:
        name: Parameter identifier.
        description: Human-readable explanation.
        param_type: The data type of the parameter.
        required: Whether the parameter must be provided.
        default: Default value when not provided (only for optional params).
    """

    name: str
    description: str
    param_type: ParameterType = ParameterType.STRING
    required: bool = True
    default: Any = None


@dataclass(frozen=True, slots=True)
class ToolCapability:
    """Declares what a tool can do, for discovery and sandboxing.

    Attributes:
        reads_filesystem: Tool may read files.
        writes_filesystem: Tool may write/create/delete files.
        network_access: Tool may make network requests.
        subprocess_access: Tool may spawn subprocesses.
    """

    reads_filesystem: bool = False
    writes_filesystem: bool = False
    network_access: bool = False
    subprocess_access: bool = False


@dataclass(slots=True)
class ToolResult:
    """The outcome of a tool execution.

    Attributes:
        success: Whether the tool completed without error.
        data: The result payload (any serialisable value).
        error: Human-readable error message on failure.
        metadata: Optional extra info (timing, source, etc.).
    """

    success: bool
    data: Any = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # --- convenience constructors ---

    @classmethod
    def ok(cls, data: Any = None, **metadata: Any) -> ToolResult:
        """Create a successful result."""
        return cls(success=True, data=data, metadata=metadata)

    @classmethod
    def fail(cls, error: str, **metadata: Any) -> ToolResult:
        """Create a failed result."""
        return cls(success=False, error=error, metadata=metadata)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict."""
        out: dict[str, Any] = {"success": self.success}
        if self.data is not None:
            out["data"] = self.data
        if self.error is not None:
            out["error"] = self.error
        if self.metadata:
            out["metadata"] = self.metadata
        return out


class Tool(ABC):
    """Abstract base class every Yoda tool must implement.

    Subclasses declare metadata via class attributes and implement
    the async ``execute`` method.
    """

    # --- metadata (override in subclasses) ---

    name: str
    """Unique tool identifier (snake_case)."""

    description: str
    """One-line human-readable summary of what the tool does."""

    parameters: list[ToolParameter] = []
    """Ordered list of parameters the tool accepts."""

    capabilities: ToolCapability = ToolCapability()
    """Declared capabilities for sandboxing / permission checks."""

    # --- validation ---

    def validate_params(self, kwargs: dict[str, Any]) -> list[str]:
        """Validate *kwargs* against declared parameters.

        Returns a list of validation error strings (empty = valid).
        """
        errors: list[str] = []
        param_map = {p.name: p for p in self.parameters}

        # Check required params are present
        for param in self.parameters:
            if param.required and param.name not in kwargs:
                errors.append(f"Missing required parameter: '{param.name}'")

        # Check for unknown params
        for key in kwargs:
            if key not in param_map:
                errors.append(f"Unknown parameter: '{key}'")

        return errors

    def apply_defaults(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Return a copy of *kwargs* with defaults filled in."""
        result = dict(kwargs)
        for param in self.parameters:
            if param.name not in result and param.default is not None:
                result[param.name] = param.default
        return result

    # --- execution ---

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Run the tool with the given keyword arguments.

        Implementations must return a ``ToolResult``.
        """
        ...

    # --- introspection ---

    def manifest(self) -> dict[str, Any]:
        """Return a JSON-serialisable manifest describing this tool."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": [
                {
                    "name": p.name,
                    "description": p.description,
                    "type": p.param_type.value,
                    "required": p.required,
                    **({"default": p.default} if p.default is not None else {}),
                }
                for p in self.parameters
            ],
            "capabilities": {
                "reads_filesystem": self.capabilities.reads_filesystem,
                "writes_filesystem": self.capabilities.writes_filesystem,
                "network_access": self.capabilities.network_access,
                "subprocess_access": self.capabilities.subprocess_access,
            },
        }

    def __repr__(self) -> str:
        return f"<Tool {self.name!r}>"
