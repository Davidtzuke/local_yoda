"""Computer access tools: screenshot, OCR, mouse/keyboard control, app launcher."""

from yoda.tools.computer.screen import register_screen_tools
from yoda.tools.computer.input_control import register_input_tools
from yoda.tools.computer.app_launcher import register_app_tools

__all__ = [
    "register_screen_tools",
    "register_input_tools",
    "register_app_tools",
]


def register_all_computer_tools() -> None:
    """Import all computer tool modules to trigger @tool decorator registration."""
    import yoda.tools.computer.screen  # noqa: F401
    import yoda.tools.computer.input_control  # noqa: F401
    import yoda.tools.computer.app_launcher  # noqa: F401
