"""Shell command execution tools with safety controls."""

from __future__ import annotations

import asyncio
import os
import shlex
from pathlib import Path

from yoda.tools.registry import ToolPermission, tool

# Commands that are always blocked
_BLOCKED_COMMANDS = {
    "rm -rf /",
    "rm -rf /*",
    "mkfs",
    "dd if=/dev/zero",
    ":(){:|:&};:",
    "format",
}

# Patterns that trigger approval
_DANGEROUS_PATTERNS = [
    "rm -rf",
    "sudo",
    "chmod 777",
    "curl | sh",
    "wget | sh",
    "eval(",
    "exec(",
    "> /dev/sd",
]


def _is_blocked(command: str) -> bool:
    """Check if a command is in the blocklist."""
    cmd_lower = command.strip().lower()
    return any(blocked in cmd_lower for blocked in _BLOCKED_COMMANDS)


def _is_dangerous(command: str) -> bool:
    """Check if a command contains dangerous patterns."""
    cmd_lower = command.lower()
    return any(p in cmd_lower for p in _DANGEROUS_PATTERNS)


def register_shell_tools() -> None:
    """No-op — importing this module registers tools via decorators."""


@tool(
    name="run_command",
    permission=ToolPermission.EXECUTE,
    category="shell",
    tags=["shell", "command", "execute"],
    timeout=60.0,
    retries=1,
)
async def run_command(
    command: str,
    cwd: str = ".",
    timeout: float = 30.0,
    env: dict[str, str] | None = None,
) -> str:
    """Execute a shell command and return stdout/stderr.

    Args:
        command: Shell command to execute.
        cwd: Working directory (default: current).
        timeout: Timeout in seconds.
        env: Additional environment variables.
    """
    if _is_blocked(command):
        raise PermissionError(f"Blocked dangerous command: {command}")

    if _is_dangerous(command):
        # The executor's approval flow will handle this
        pass

    work_dir = Path(cwd).expanduser().resolve()
    if not work_dir.is_dir():
        raise ValueError(f"Working directory not found: {work_dir}")

    full_env = {**os.environ, **(env or {})}

    proc = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(work_dir),
        env=full_env,
    )

    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise TimeoutError(f"Command timed out after {timeout}s: {command}")

    output_parts: list[str] = []
    if stdout:
        output_parts.append(stdout.decode(errors="replace"))
    if stderr:
        output_parts.append(f"[stderr]\n{stderr.decode(errors='replace')}")

    output = "\n".join(output_parts).strip()

    # Cap output length
    if len(output) > 50_000:
        output = output[:50_000] + f"\n... (truncated, {len(output)} total chars)"

    exit_info = f"[exit code: {proc.returncode}]"
    return f"{output}\n{exit_info}" if output else exit_info


@tool(
    name="run_python",
    permission=ToolPermission.EXECUTE,
    category="shell",
    tags=["python", "execute", "code"],
    timeout=60.0,
)
async def run_python(code: str, timeout: float = 30.0) -> str:
    """Execute Python code and return the output.

    Args:
        code: Python code to execute.
        timeout: Timeout in seconds.
    """
    proc = await asyncio.create_subprocess_exec(
        "python3", "-c", code,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise TimeoutError(f"Python execution timed out after {timeout}s")

    output = ""
    if stdout:
        output += stdout.decode(errors="replace")
    if stderr:
        output += f"\n[stderr]\n{stderr.decode(errors='replace')}"

    return output.strip() or f"[exit code: {proc.returncode}]"


@tool(
    name="get_env",
    permission=ToolPermission.READ,
    category="shell",
    tags=["environment", "variable"],
    timeout=5.0,
)
async def get_env(name: str, default: str = "") -> str:
    """Get an environment variable value.

    Args:
        name: Environment variable name.
        default: Default value if not set.
    """
    return os.environ.get(name, default)
