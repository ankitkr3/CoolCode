"""File and git event watchers for the daemon.

Monitors the project directory for changes and emits events.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import AsyncIterator

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    FILE_CHANGED = "file_changed"
    FILE_CREATED = "file_created"
    FILE_DELETED = "file_deleted"
    GIT_COMMIT = "git_commit"
    GIT_BRANCH_SWITCH = "git_branch_switch"
    TEST_FAILED = "test_failed"


@dataclass
class WatchEvent:
    """A filesystem or git event detected by a watcher."""

    event_type: EventType
    path: str
    detail: str = ""
    timestamp: float = field(default_factory=time.time)


# File patterns to watch
DEFAULT_WATCH_PATTERNS = {"*.py", "*.js", "*.ts", "*.tsx", "*.jsx", "*.go", "*.rs", "*.java", "*.rb"}
IGNORE_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", ".coolcode", ".pytest_cache", "dist", "build"}


class FileWatcher:
    """Watches for file changes using polling (no external deps).

    Uses modification time tracking — lightweight and cross-platform.
    For production use, can be upgraded to watchfiles/inotify.
    """

    def __init__(
        self,
        project_dir: str,
        patterns: set[str] | None = None,
        poll_interval: float = 2.0,
    ):
        self.project_dir = Path(project_dir)
        self.patterns = patterns or DEFAULT_WATCH_PATTERNS
        self.poll_interval = poll_interval
        self._file_mtimes: dict[str, float] = {}
        self._running = False

    def _should_watch(self, path: Path) -> bool:
        """Check if a file matches watch patterns and isn't in an ignored dir."""
        # Check ignore dirs
        for part in path.parts:
            if part in IGNORE_DIRS:
                return False
        # Check patterns
        return any(path.match(p) for p in self.patterns)

    def _scan(self) -> dict[str, float]:
        """Scan project directory and return {path: mtime} dict."""
        mtimes: dict[str, float] = {}
        try:
            for root, dirs, files in os.walk(self.project_dir):
                # Skip ignored directories
                dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
                for f in files:
                    path = Path(root) / f
                    if self._should_watch(path):
                        try:
                            mtimes[str(path)] = path.stat().st_mtime
                        except (OSError, FileNotFoundError):
                            pass
        except Exception as e:
            logger.warning(f"File scan error: {e}")
        return mtimes

    async def watch(self) -> AsyncIterator[WatchEvent]:
        """Yield file change events. Runs forever until stopped."""
        self._running = True
        self._file_mtimes = self._scan()

        while self._running:
            await asyncio.sleep(self.poll_interval)

            new_mtimes = self._scan()

            # Detect changes
            for path, mtime in new_mtimes.items():
                if path not in self._file_mtimes:
                    yield WatchEvent(EventType.FILE_CREATED, path, f"new file: {Path(path).name}")
                elif mtime != self._file_mtimes[path]:
                    yield WatchEvent(EventType.FILE_CHANGED, path, f"modified: {Path(path).name}")

            # Detect deletions
            for path in set(self._file_mtimes) - set(new_mtimes):
                yield WatchEvent(EventType.FILE_DELETED, path, f"deleted: {Path(path).name}")

            self._file_mtimes = new_mtimes

    def stop(self) -> None:
        self._running = False


class GitWatcher:
    """Watches for git events by polling .git directory.

    Detects branch switches and new commits.
    """

    def __init__(self, project_dir: str, poll_interval: float = 5.0):
        self.project_dir = Path(project_dir)
        self.poll_interval = poll_interval
        self._running = False
        self._last_head = ""
        self._last_commit = ""

    def _read_head(self) -> str:
        """Read current HEAD reference."""
        head_file = self.project_dir / ".git" / "HEAD"
        try:
            return head_file.read_text().strip()
        except (OSError, FileNotFoundError):
            return ""

    def _read_commit(self) -> str:
        """Read current commit hash."""
        from coolcode.tools.git import _run_git
        result = _run_git(["rev-parse", "HEAD"], cwd=str(self.project_dir))
        return result.strip().split("\n")[0] if result and "Error" not in result else ""

    async def watch(self) -> AsyncIterator[WatchEvent]:
        """Yield git events. Runs forever until stopped."""
        self._running = True
        self._last_head = self._read_head()
        self._last_commit = self._read_commit()

        while self._running:
            await asyncio.sleep(self.poll_interval)

            current_head = self._read_head()
            current_commit = self._read_commit()

            if current_head != self._last_head:
                yield WatchEvent(
                    EventType.GIT_BRANCH_SWITCH,
                    str(self.project_dir / ".git" / "HEAD"),
                    f"branch: {current_head}",
                )
                self._last_head = current_head

            if current_commit and current_commit != self._last_commit:
                yield WatchEvent(
                    EventType.GIT_COMMIT,
                    str(self.project_dir),
                    f"new commit: {current_commit[:8]}",
                )
                self._last_commit = current_commit

    def stop(self) -> None:
        self._running = False
