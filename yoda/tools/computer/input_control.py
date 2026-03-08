"""Mouse and keyboard control tools via pyautogui."""

from __future__ import annotations

import asyncio
import logging
import time

from yoda.tools.registry import ToolPermission, tool

logger = logging.getLogger(__name__)


def register_input_tools() -> None:
    """No-op — importing this module registers tools via decorators."""


def _ensure_pyautogui():
    """Import and configure pyautogui with safety settings."""
    import pyautogui
    # Enable fail-safe: move mouse to corner to abort
    pyautogui.FAILSAFE = True
    # Small pause between actions for safety
    pyautogui.PAUSE = 0.1
    return pyautogui


@tool(
    name="mouse_click",
    permission=ToolPermission.COMPUTER,
    category="computer",
    tags=["mouse", "click", "input"],
    requires_approval=True,
    timeout=10.0,
)
async def mouse_click(
    x: int,
    y: int,
    button: str = "left",
    clicks: int = 1,
    interval: float = 0.1,
) -> str:
    """Click the mouse at a screen position.

    Args:
        x: X coordinate.
        y: Y coordinate.
        button: Mouse button (left, right, middle).
        clicks: Number of clicks.
        interval: Interval between clicks in seconds.
    """
    pyautogui = _ensure_pyautogui()
    pyautogui.click(x=x, y=y, button=button, clicks=clicks, interval=interval)
    return f"Clicked {button} at ({x}, {y}) x{clicks}"


@tool(
    name="mouse_move",
    permission=ToolPermission.COMPUTER,
    category="computer",
    tags=["mouse", "move", "input"],
    timeout=10.0,
)
async def mouse_move(x: int, y: int, duration: float = 0.25) -> str:
    """Move the mouse to a screen position.

    Args:
        x: Target X coordinate.
        y: Target Y coordinate.
        duration: Movement duration in seconds.
    """
    pyautogui = _ensure_pyautogui()
    pyautogui.moveTo(x=x, y=y, duration=duration)
    return f"Mouse moved to ({x}, {y})"


@tool(
    name="mouse_drag",
    permission=ToolPermission.COMPUTER,
    category="computer",
    tags=["mouse", "drag", "input"],
    requires_approval=True,
    timeout=10.0,
)
async def mouse_drag(
    start_x: int,
    start_y: int,
    end_x: int,
    end_y: int,
    duration: float = 0.5,
    button: str = "left",
) -> str:
    """Drag the mouse from one position to another.

    Args:
        start_x: Start X coordinate.
        start_y: Start Y coordinate.
        end_x: End X coordinate.
        end_y: End Y coordinate.
        duration: Drag duration in seconds.
        button: Mouse button to hold during drag.
    """
    pyautogui = _ensure_pyautogui()
    pyautogui.moveTo(start_x, start_y)
    pyautogui.drag(
        end_x - start_x,
        end_y - start_y,
        duration=duration,
        button=button,
    )
    return f"Dragged from ({start_x},{start_y}) to ({end_x},{end_y})"


@tool(
    name="mouse_scroll",
    permission=ToolPermission.COMPUTER,
    category="computer",
    tags=["mouse", "scroll", "input"],
    timeout=10.0,
)
async def mouse_scroll(amount: int, x: int = 0, y: int = 0) -> str:
    """Scroll the mouse wheel.

    Args:
        amount: Scroll amount (positive=up, negative=down).
        x: X position to scroll at (0 = current).
        y: Y position to scroll at (0 = current).
    """
    pyautogui = _ensure_pyautogui()
    if x and y:
        pyautogui.scroll(amount, x=x, y=y)
    else:
        pyautogui.scroll(amount)
    direction = "up" if amount > 0 else "down"
    return f"Scrolled {direction} by {abs(amount)}"


@tool(
    name="keyboard_type",
    permission=ToolPermission.COMPUTER,
    category="computer",
    tags=["keyboard", "type", "input"],
    requires_approval=True,
    timeout=30.0,
)
async def keyboard_type(text: str, interval: float = 0.02) -> str:
    """Type text using the keyboard.

    Args:
        text: Text to type.
        interval: Interval between keystrokes in seconds.
    """
    pyautogui = _ensure_pyautogui()
    pyautogui.typewrite(text, interval=interval)
    return f"Typed {len(text)} characters"


@tool(
    name="keyboard_hotkey",
    permission=ToolPermission.COMPUTER,
    category="computer",
    tags=["keyboard", "hotkey", "shortcut", "input"],
    requires_approval=True,
    timeout=10.0,
)
async def keyboard_hotkey(keys: str) -> str:
    """Press a keyboard shortcut (e.g., 'ctrl+c', 'cmd+shift+s').

    Args:
        keys: Key combination separated by '+' (e.g., 'ctrl+c', 'alt+tab').
    """
    pyautogui = _ensure_pyautogui()
    key_list = [k.strip() for k in keys.split("+")]
    pyautogui.hotkey(*key_list)
    return f"Pressed hotkey: {keys}"


@tool(
    name="keyboard_press",
    permission=ToolPermission.COMPUTER,
    category="computer",
    tags=["keyboard", "press", "key", "input"],
    timeout=10.0,
)
async def keyboard_press(key: str, presses: int = 1) -> str:
    """Press a single key.

    Args:
        key: Key name (enter, tab, escape, space, backspace, delete, up, down, left, right, etc.).
        presses: Number of times to press.
    """
    pyautogui = _ensure_pyautogui()
    pyautogui.press(key, presses=presses)
    return f"Pressed '{key}' x{presses}"


@tool(
    name="mouse_position",
    permission=ToolPermission.COMPUTER,
    category="computer",
    tags=["mouse", "position"],
    timeout=5.0,
)
async def mouse_position() -> str:
    """Get the current mouse cursor position."""
    pyautogui = _ensure_pyautogui()
    pos = pyautogui.position()
    return f"Mouse position: ({pos.x}, {pos.y})"
