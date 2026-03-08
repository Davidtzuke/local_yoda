"""Built-in tools for Yoda: file ops, shell, web, calendar, notes."""

from yoda.tools.builtins.file_ops import register_file_tools
from yoda.tools.builtins.shell import register_shell_tools
from yoda.tools.builtins.web import register_web_tools
from yoda.tools.builtins.calendar_tool import register_calendar_tools
from yoda.tools.builtins.notes import register_notes_tools

__all__ = [
    "register_file_tools",
    "register_shell_tools",
    "register_web_tools",
    "register_calendar_tools",
    "register_notes_tools",
]


def register_all_builtins() -> None:
    """Import all built-in tool modules to trigger @tool decorator registration."""
    # Importing the modules triggers the @tool decorators
    import yoda.tools.builtins.file_ops  # noqa: F401
    import yoda.tools.builtins.shell  # noqa: F401
    import yoda.tools.builtins.web  # noqa: F401
    import yoda.tools.builtins.calendar_tool  # noqa: F401
    import yoda.tools.builtins.notes  # noqa: F401
