"""Git operation tools."""

from __future__ import annotations

import subprocess
from pathlib import Path


def _run_git(args: list[str], cwd: str = ".") -> str:
    """Run a git command and return output."""
    try:
        result = subprocess.run(
            ["git"] + args,
            capture_output=True,
            text=True,
            cwd=Path(cwd).resolve(),
            timeout=30,
        )
        output = result.stdout
        if result.stderr:
            output += f"\n[stderr] {result.stderr}"
        if result.returncode != 0:
            output += f"\n[exit code: {result.returncode}]"
        return output.strip() or "(no output)"
    except FileNotFoundError:
        return "Error: git is not installed"
    except subprocess.TimeoutExpired:
        return "Error: git command timed out"
    except Exception as e:
        return f"Error: {e}"


def git_status(directory: str = ".") -> str:
    """Show the working tree status.

    Args:
        directory: Repository directory.

    Returns:
        Git status output.
    """
    return _run_git(["status", "--short", "--branch"], cwd=directory)


def git_diff(directory: str = ".", staged: bool = False, file_path: str = "") -> str:
    """Show changes in the working tree or staging area.

    Args:
        directory: Repository directory.
        staged: If True, show staged changes only.
        file_path: Optional specific file to diff.

    Returns:
        Git diff output.
    """
    args = ["diff"]
    if staged:
        args.append("--cached")
    if file_path:
        args.extend(["--", file_path])
    return _run_git(args, cwd=directory)


def git_log(directory: str = ".", count: int = 10) -> str:
    """Show recent commit history.

    Args:
        directory: Repository directory.
        count: Number of commits to show.

    Returns:
        Git log output.
    """
    return _run_git(
        ["log", f"-{count}", "--oneline", "--graph", "--decorate"],
        cwd=directory,
    )


def git_commit(message: str, directory: str = ".", files: str = "") -> str:
    """Stage and commit changes.

    Args:
        message: Commit message.
        directory: Repository directory.
        files: Space-separated file paths to stage. If empty, stages all changes.

    Returns:
        Commit result.
    """
    # Stage
    if files:
        for f in files.split():
            result = _run_git(["add", f], cwd=directory)
            if "Error" in result:
                return result
    else:
        result = _run_git(["add", "-A"], cwd=directory)
        if "Error" in result:
            return result

    # Commit
    return _run_git(["commit", "-m", message], cwd=directory)
