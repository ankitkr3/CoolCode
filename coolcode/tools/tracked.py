"""Tracked versions of tools that emit status updates when called."""

from __future__ import annotations

import functools
import threading
from pathlib import Path
from typing import Any, Callable

from coolcode.status import StatusTracker


class FileReadCache:
    """Thread-safe shared cache for file reads across workers in a goal pipeline.

    Prevents multiple workers from re-reading the same file independently.
    Keyed by (resolved_path, offset, limit).
    """

    def __init__(self) -> None:
        self._cache: dict[tuple[str, int, int], str] = {}
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, path: str, offset: int, limit: int) -> str | None:
        key = (path, offset, limit)
        with self._lock:
            result = self._cache.get(key)
            if result is not None:
                self.hits += 1
            return result

    def put(self, path: str, offset: int, limit: int, content: str) -> None:
        key = (path, offset, limit)
        with self._lock:
            self._cache[key] = content
            self.misses += 1

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self.hits = 0
            self.misses = 0


def make_tracked_tools(
    status: StatusTracker,
    worker_id: str = "agent",
    file_cache: FileReadCache | None = None,
) -> list[Callable]:
    """Create tool functions that emit status updates when invoked.

    These wrap the original tools so the CLI can show real-time activity like:
      "reading src/main.py (245 lines)"
      "searching for 'def handle_request' in **/*.py"
      "listing /Users/ankit/project/src (12 entries)"
      "editing config.py"
      "running: npm test"
    """

    def _emit(action: str, detail: str = "") -> None:
        status.emit(worker_id, action, detail)

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
        resolved = str(path)

        # Check shared cache first (avoids duplicate reads across parallel workers)
        if file_cache is not None:
            cached = file_cache.get(resolved, offset, limit)
            if cached is not None:
                lines_count = cached.count("\n")
                _emit("read (cached)", f"{path.name} ({lines_count} lines)")
                return cached

        _emit("reading", f"{path.name}")
        from coolcode.tools.files import read_file as _read
        result = _read(file_path, offset, limit)
        lines_count = result.count("\n")
        _emit("read", f"{path.name} ({lines_count} lines)")

        # Store in shared cache
        if file_cache is not None:
            file_cache.put(resolved, offset, limit, result)

        return result

    def write_file(file_path: str, content: str) -> str:
        """Write content to a file, creating parent directories if needed.

        Args:
            file_path: Absolute or relative path to the file.
            content: The content to write.

        Returns:
            Success or error message.
        """
        path = Path(file_path).resolve()
        _emit("writing", f"{path.name} ({len(content)} bytes)")
        from coolcode.tools.files import write_file as _write
        return _write(file_path, content)

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
        _emit("editing", f"{path.name}")
        from coolcode.tools.files import edit_file as _edit
        return _edit(file_path, old_string, new_string)

    def list_dir(directory: str = ".") -> str:
        """List files and directories in the given path.

        Args:
            directory: Path to list. Default is current directory.

        Returns:
            Formatted directory listing.
        """
        path = Path(directory).resolve()
        _emit("listing", f"{path.name}/")
        from coolcode.tools.files import list_dir as _list
        result = _list(directory)
        count = result.count("\n")
        _emit("listed", f"{path.name}/ ({count} entries)")
        return result

    def glob_search(pattern: str, directory: str = ".") -> str:
        """Search for files matching a glob pattern.

        Args:
            pattern: Glob pattern (e.g., '**/*.py', 'src/**/*.ts').
            directory: Root directory to search from.

        Returns:
            List of matching file paths.
        """
        _emit("searching", f"glob: {pattern}")
        from coolcode.tools.files import glob_search as _glob
        result = _glob(pattern, directory)
        match_line = result.split("\n")[0] if result else ""
        _emit("found", match_line[:80])
        return result

    def execute_shell(command: str, working_dir: str = ".", timeout: int = 120) -> str:
        """Execute a shell command and return its output.

        Args:
            command: The shell command to execute.
            working_dir: Working directory for the command.
            timeout: Timeout in seconds (default 120).

        Returns:
            Command output (stdout + stderr) or error message.
        """
        # Show abbreviated command
        display_cmd = command[:60] + "..." if len(command) > 60 else command
        _emit("running", display_cmd)
        from coolcode.tools.shell import execute_shell as _exec
        result = _exec(command, working_dir, timeout)
        lines = result.count("\n")
        _emit("ran", f"{display_cmd} ({lines} lines output)")
        return result

    def grep_search(
        pattern: str,
        directory: str = ".",
        file_glob: str = "",
        max_results: int = 50,
        case_insensitive: bool = False,
    ) -> str:
        """Search file contents using regex pattern.

        Args:
            pattern: Regex pattern to search for.
            directory: Root directory to search in.
            file_glob: Optional glob to filter files (e.g., '*.py').
            max_results: Maximum number of results to return.
            case_insensitive: Whether to ignore case.

        Returns:
            Matching lines with file paths and line numbers.
        """
        detail = f"'{pattern}'"
        if file_glob:
            detail += f" in {file_glob}"
        _emit("grepping", detail)
        from coolcode.tools.search import grep_search as _grep
        result = _grep(pattern, directory, file_glob, max_results, case_insensitive)
        matches = result.count("\n") if result else 0
        _emit("grep done", f"'{pattern}' ({matches} matches)")
        return result

    def find_definition(name: str, directory: str = ".", language: str = "") -> str:
        """Find the definition of a class, function, or variable in the codebase.

        Args:
            name: The name to search for.
            directory: Root directory to search in.
            language: Optional language hint.

        Returns:
            File paths and lines where the definition is found.
        """
        _emit("finding", f"definition of '{name}'")
        from coolcode.tools.search import find_definition as _find
        result = _find(name, directory, language)
        _emit("found def", f"'{name}'")
        return result

    def git_status(directory: str = ".") -> str:
        """Show the working tree status.

        Args:
            directory: Repository directory.

        Returns:
            Git status output.
        """
        _emit("git", "checking status")
        from coolcode.tools.git import git_status as _status
        return _status(directory)

    def git_diff(directory: str = ".", staged: bool = False, file_path: str = "") -> str:
        """Show changes in the working tree or staging area.

        Args:
            directory: Repository directory.
            staged: If True, show staged changes only.
            file_path: Optional specific file to diff.

        Returns:
            Git diff output.
        """
        _emit("git", f"diff {'(staged)' if staged else ''} {file_path or ''}".strip())
        from coolcode.tools.git import git_diff as _diff
        return _diff(directory, staged, file_path)

    def git_log(directory: str = ".", count: int = 10) -> str:
        """Show recent commit history.

        Args:
            directory: Repository directory.
            count: Number of commits to show.

        Returns:
            Git log output.
        """
        _emit("git", f"log (last {count})")
        from coolcode.tools.git import git_log as _log
        return _log(directory, count)

    def git_commit(message: str, directory: str = ".", files: str = "") -> str:
        """Stage and commit changes.

        Args:
            message: Commit message.
            directory: Repository directory.
            files: Space-separated file paths to stage.

        Returns:
            Commit result.
        """
        _emit("git", f"committing: {message[:50]}")
        from coolcode.tools.git import git_commit as _commit
        return _commit(message, directory, files)

    return [
        read_file,
        write_file,
        edit_file,
        list_dir,
        glob_search,
        execute_shell,
        grep_search,
        find_definition,
        git_status,
        git_diff,
        git_log,
        git_commit,
    ]
