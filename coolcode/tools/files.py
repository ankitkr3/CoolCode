"""File operation tools for Cool Code agents."""

from __future__ import annotations

import ast
import fnmatch
import json
import os
from pathlib import Path


def _validate_content(path: Path, content: str) -> str | None:
    """Validate file content by extension. Returns error message or None if OK.

    Prevents workers from writing broken code (unterminated strings, bad JSON, etc).
    """
    suffix = path.suffix.lower()

    if suffix == ".py":
        try:
            ast.parse(content, filename=str(path))
        except SyntaxError as e:
            preview = content[:300] + ("..." if len(content) > 300 else "")
            return (
                f"SyntaxError in {path.name} at line {e.lineno}: {e.msg}\n"
                f"Preview: {preview}\n"
                f"[WRITE BLOCKED] Fix the syntax error and retry."
            )

    elif suffix == ".json":
        try:
            json.loads(content)
        except json.JSONDecodeError as e:
            return (
                f"Invalid JSON in {path.name}: {e.msg} at line {e.lineno} col {e.colno}\n"
                f"[WRITE BLOCKED] Fix the JSON and retry."
            )

    elif suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
            yaml.safe_load(content)
        except ImportError:
            pass  # PyYAML not installed — skip validation
        except Exception as e:
            return f"Invalid YAML in {path.name}: {e}\n[WRITE BLOCKED]"

    elif suffix == ".toml":
        try:
            import tomllib  # py311+
            tomllib.loads(content)
        except ImportError:
            try:
                import tomli  # type: ignore
                tomli.loads(content)
            except ImportError:
                pass
            except Exception as e:
                return f"Invalid TOML in {path.name}: {e}\n[WRITE BLOCKED]"
        except Exception as e:
            return f"Invalid TOML in {path.name}: {e}\n[WRITE BLOCKED]"

    return None


def _atomic_write(path: Path, content: str) -> None:
    """Write content to path atomically via a temp file + rename.

    Prevents partial/corrupt files if a worker crashes mid-write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    try:
        tmp.write_text(content, encoding="utf-8")
        os.replace(tmp, path)  # atomic on POSIX and Windows
    except Exception:
        # Clean up temp file on failure
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        raise


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

    Validates Python/JSON/YAML/TOML syntax before writing. Writes atomically
    via a .tmp file + rename so crashes don't leave half-written files.

    Args:
        file_path: Absolute or relative path to the file.
        content: The content to write.

    Returns:
        Success or error message.
    """
    path = Path(file_path).resolve()

    # Validate syntax for known file types — block write on error
    error = _validate_content(path, content)
    if error:
        return f"Error: {error}"

    try:
        _atomic_write(path, content)
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

        # Validate syntax after the edit
        error = _validate_content(path, new_text)
        if error:
            return f"Error: edit would produce invalid file — {error}"

        _atomic_write(path, new_text)
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
