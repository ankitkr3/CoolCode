"""Shell execution tool with safety checks."""

from __future__ import annotations

import subprocess
from pathlib import Path

# Commands that are always blocked
BLOCKED_COMMANDS = {
    "rm -rf /", "rm -rf /*", "mkfs", "dd if=", ":(){", "fork bomb",
    "chmod -R 777 /", "shutdown", "reboot", "halt", "poweroff",
}


def execute_shell(
    command: str,
    working_dir: str = ".",
    timeout: int = 120,
) -> str:
    """Execute a shell command and return its output.

    Args:
        command: The shell command to execute.
        working_dir: Working directory for the command.
        timeout: Timeout in seconds (default 120).

    Returns:
        Command output (stdout + stderr) or error message.
    """
    # Safety check
    cmd_lower = command.lower().strip()
    for blocked in BLOCKED_COMMANDS:
        if blocked in cmd_lower:
            return f"Error: Blocked dangerous command: {command}"

    cwd = Path(working_dir).resolve()
    if not cwd.exists():
        return f"Error: Working directory not found: {cwd}"

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=str(cwd),
            timeout=timeout,
            env=None,  # inherit environment
        )
        output_parts = []
        if result.stdout:
            output_parts.append(result.stdout)
        if result.stderr:
            output_parts.append(f"[stderr]\n{result.stderr}")
        if result.returncode != 0:
            output_parts.append(f"[exit code: {result.returncode}]")

        output = "\n".join(output_parts)
        # Truncate very long outputs
        if len(output) > 50000:
            output = output[:50000] + f"\n... (truncated, {len(output)} total chars)"
        return output or "(no output)"

    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {timeout}s: {command}"
    except Exception as e:
        return f"Error executing command: {e}"
