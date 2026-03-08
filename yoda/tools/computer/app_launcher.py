"""Application launcher and window management tools."""

from __future__ import annotations

import asyncio
import logging
import platform
import subprocess
from pathlib import Path

from yoda.tools.registry import ToolPermission, tool

logger = logging.getLogger(__name__)

_SYSTEM = platform.system().lower()


def register_app_tools() -> None:
    """No-op — importing this module registers tools via decorators."""


@tool(
    name="open_application",
    permission=ToolPermission.EXECUTE,
    category="computer",
    tags=["app", "launch", "open"],
    requires_approval=True,
    timeout=15.0,
)
async def open_application(name: str, args: str = "") -> str:
    """Open an application by name.

    Args:
        name: Application name or path (e.g., 'firefox', 'code', '/usr/bin/vim').
        args: Command-line arguments to pass.
    """
    arg_list = args.split() if args else []

    if _SYSTEM == "darwin":
        # macOS: try 'open -a' first
        proc = await asyncio.create_subprocess_exec(
            "open", "-a", name, *arg_list,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode == 0:
            return f"Opened {name}"
        # Fallback to direct execution
        proc = await asyncio.create_subprocess_exec(
            name, *arg_list,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        return f"Launched {name} (pid: {proc.pid})"

    elif _SYSTEM == "linux":
        # Linux: try xdg-open for files/URLs, otherwise direct
        if name.startswith(("http://", "https://", "/")) or "." in name:
            proc = await asyncio.create_subprocess_exec(
                "xdg-open", name, *arg_list,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        else:
            proc = await asyncio.create_subprocess_exec(
                name, *arg_list,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        return f"Launched {name} (pid: {proc.pid})"

    elif _SYSTEM == "windows":
        proc = await asyncio.create_subprocess_exec(
            "cmd", "/c", "start", "", name, *arg_list,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        return f"Opened {name}"

    return f"Unsupported platform: {_SYSTEM}"


@tool(
    name="open_url",
    permission=ToolPermission.EXECUTE,
    category="computer",
    tags=["browser", "url", "web"],
    timeout=10.0,
)
async def open_url(url: str, browser: str = "") -> str:
    """Open a URL in the default or specified browser.

    Args:
        url: URL to open.
        browser: Specific browser to use (optional).
    """
    import webbrowser

    if browser:
        try:
            b = webbrowser.get(browser)
            b.open(url)
        except webbrowser.Error:
            webbrowser.open(url)
    else:
        webbrowser.open(url)
    return f"Opened URL: {url}"


@tool(
    name="list_processes",
    permission=ToolPermission.READ,
    category="computer",
    tags=["process", "list", "system"],
    timeout=10.0,
)
async def list_processes(filter_name: str = "", limit: int = 30) -> str:
    """List running processes.

    Args:
        filter_name: Filter by process name (optional).
        limit: Maximum number of processes to show.
    """
    if _SYSTEM == "windows":
        cmd = "tasklist"
    else:
        cmd = "ps aux"

    proc = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await proc.communicate()
    output = stdout.decode(errors="replace")

    lines = output.strip().split("\n")
    if filter_name:
        header = lines[0] if lines else ""
        filtered = [l for l in lines[1:] if filter_name.lower() in l.lower()]
        lines = [header] + filtered

    return "\n".join(lines[:limit + 1])


@tool(
    name="clipboard_read",
    permission=ToolPermission.READ,
    category="computer",
    tags=["clipboard", "paste", "read"],
    timeout=5.0,
)
async def clipboard_read() -> str:
    """Read text from the system clipboard."""
    try:
        import pyperclip
        return pyperclip.paste() or "(clipboard empty)"
    except ImportError:
        # Fallback for macOS
        if _SYSTEM == "darwin":
            proc = await asyncio.create_subprocess_exec(
                "pbpaste",
                stdout=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
            return stdout.decode() or "(clipboard empty)"
        elif _SYSTEM == "linux":
            try:
                proc = await asyncio.create_subprocess_exec(
                    "xclip", "-selection", "clipboard", "-o",
                    stdout=asyncio.subprocess.PIPE,
                )
                stdout, _ = await proc.communicate()
                return stdout.decode() or "(clipboard empty)"
            except FileNotFoundError:
                return "Error: Install xclip or pyperclip"
        return "Error: pyperclip not installed"


@tool(
    name="clipboard_write",
    permission=ToolPermission.WRITE,
    category="computer",
    tags=["clipboard", "copy", "write"],
    timeout=5.0,
)
async def clipboard_write(text: str) -> str:
    """Write text to the system clipboard.

    Args:
        text: Text to copy to clipboard.
    """
    try:
        import pyperclip
        pyperclip.copy(text)
        return f"Copied {len(text)} chars to clipboard"
    except ImportError:
        if _SYSTEM == "darwin":
            proc = await asyncio.create_subprocess_exec(
                "pbcopy",
                stdin=asyncio.subprocess.PIPE,
            )
            await proc.communicate(text.encode())
            return f"Copied {len(text)} chars to clipboard"
        elif _SYSTEM == "linux":
            try:
                proc = await asyncio.create_subprocess_exec(
                    "xclip", "-selection", "clipboard",
                    stdin=asyncio.subprocess.PIPE,
                )
                await proc.communicate(text.encode())
                return f"Copied {len(text)} chars to clipboard"
            except FileNotFoundError:
                return "Error: Install xclip or pyperclip"
        return "Error: pyperclip not installed"
