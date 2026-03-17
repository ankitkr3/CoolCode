"""File operation tools for Cool Code agents."""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path


def read_file(file_path: str, offset: int = 0, limit: int = 2000) -> str:
    """Read a file's contents with optional line range.

    Args:
        file_path: Absolute or relative path to the file.
        offset: Starting line number (0-based). Default 0.
        limit: Maximum number of lines to return. Default 2000.

    Returns:
        File contents with line numbers prefixed.
    """
    path = Path(file_path).resolve()
    if not path.exists():
        return f"Error: File not found: {path}"
    if not path.is_file():
        return f"Error: Not a file: {path}"
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        selected = lines[offset : offset + limit]
        numbered = [f"{i + offset + 1:>6}\t{line}" for i, line in enumerate(selected)]
        header = f"# {path} ({len(lines)} lines total, showing {offset+1}-{offset+len(selected)})"
        return header + "\n" + "\n".join(numbered)
    except Exception as e:
        return f"Error reading {path}: {e}"


def write_file(file_path: str, content: str) -> str:
    """Write content to a file, creating parent directories if needed.

    Args:
        file_path: Absolute or relative path to the file.
        content: The content to write.

    Returns:
        Success or error message.
    """
    path = Path(file_path).resolve()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"Successfully wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error writing {path}: {e}"


def edit_file(file_path: str, old_string: str, new_string: str) -> str:
    """Replace an exact string in a file.

    Args:
        file_path: Path to the file to edit.
        old_string: The exact text to find and replace.
        new_string: The replacement text.

    Returns:
        Success or error message.
    """
    path = Path(file_path).resolve()
    if not path.exists():
        return f"Error: File not found: {path}"
    try:
        text = path.read_text(encoding="utf-8")
        count = text.count(old_string)
        if count == 0:
            return f"Error: old_string not found in {path}"
        if count > 1:
            return f"Error: old_string found {count} times in {path}. Provide more context to make it unique."
        new_text = text.replace(old_string, new_string, 1)
        path.write_text(new_text, encoding="utf-8")
        return f"Successfully edited {path}"
    except Exception as e:
        return f"Error editing {path}: {e}"


def list_dir(directory: str = ".") -> str:
    """List files and directories in the given path.

    Args:
        directory: Path to list. Default is current directory.

    Returns:
        Formatted directory listing.
    """
    path = Path(directory).resolve()
    if not path.exists():
        return f"Error: Directory not found: {path}"
    if not path.is_dir():
        return f"Error: Not a directory: {path}"
    try:
        entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        lines = []
        for entry in entries:
            if entry.name.startswith("."):
                continue
            prefix = "d " if entry.is_dir() else "f "
            size = ""
            if entry.is_file():
                sz = entry.stat().st_size
                if sz < 1024:
                    size = f" ({sz}B)"
                elif sz < 1024 * 1024:
                    size = f" ({sz // 1024}KB)"
                else:
                    size = f" ({sz // (1024*1024)}MB)"
            lines.append(f"  {prefix}{entry.name}{size}")
        return f"# {path}\n" + "\n".join(lines) if lines else f"# {path} (empty)"
    except Exception as e:
        return f"Error listing {path}: {e}"


def glob_search(pattern: str, directory: str = ".") -> str:
    """Search for files matching a glob pattern.

    Args:
        pattern: Glob pattern (e.g., '**/*.py', 'src/**/*.ts').
        directory: Root directory to search from.

    Returns:
        List of matching file paths.
    """
    path = Path(directory).resolve()
    try:
        matches = sorted(path.glob(pattern))
        # Limit results
        if len(matches) > 100:
            display = matches[:100]
            suffix = f"\n... and {len(matches) - 100} more"
        else:
            display = matches
            suffix = ""
        lines = [str(m.relative_to(path)) for m in display]
        return f"# glob: {pattern} in {path} ({len(matches)} matches)\n" + "\n".join(lines) + suffix
    except Exception as e:
        return f"Error in glob search: {e}"
