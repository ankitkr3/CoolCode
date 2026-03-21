"""Proactive insight engine for the daemon.

Analyzes file changes and generates insights using heuristics (no LLM needed).
"""

from __future__ import annotations

import logging
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from coolcode.daemon.watchers import EventType, WatchEvent

logger = logging.getLogger(__name__)


@dataclass
class Insight:
    """A proactive insight from the daemon."""

    message: str
    severity: str = "info"  # "info", "warning", "suggestion"
    source: str = "daemon"
    timestamp: float = field(default_factory=time.time)
    related_files: list[str] = field(default_factory=list)


class InsightEngine:
    """Generates proactive insights from file/git events.

    Uses lightweight heuristics — no LLM calls. Fast and free.
    """

    def __init__(self):
        self._change_counts: Counter = Counter()  # file -> change count
        self._change_history: list[WatchEvent] = []
        self._insights: list[Insight] = []
        self._last_insight_time: dict[str, float] = {}  # dedup by message prefix
        self._file_sizes: dict[str, int] = {}

    def analyze(self, event: WatchEvent) -> Insight | None:
        """Analyze a single event and optionally return an insight."""
        self._change_history.append(event)
        self._change_counts[event.path] += 1

        # Run heuristic checks
        insight = (
            self._check_hot_file(event)
            or self._check_test_coverage(event)
            or self._check_large_file(event)
            or self._check_rapid_changes(event)
        )

        if insight and self._should_emit(insight):
            self._insights.append(insight)
            return insight
        return None

    def _should_emit(self, insight: Insight) -> bool:
        """Dedup: don't repeat similar insights within 5 minutes."""
        prefix = insight.message[:50]
        now = time.time()
        last = self._last_insight_time.get(prefix, 0)
        if now - last < 300:  # 5 min cooldown
            return False
        self._last_insight_time[prefix] = now
        return True

    def _check_hot_file(self, event: WatchEvent) -> Insight | None:
        """Detect files being changed too frequently."""
        if event.event_type != EventType.FILE_CHANGED:
            return None
        count = self._change_counts[event.path]
        name = Path(event.path).name
        if count >= 5 and count % 5 == 0:
            return Insight(
                message=f"{name} has been modified {count} times this session — consider breaking it into smaller files",
                severity="suggestion",
                related_files=[event.path],
            )
        return None

    def _check_test_coverage(self, event: WatchEvent) -> Insight | None:
        """Detect code changes without corresponding test changes."""
        if event.event_type != EventType.FILE_CHANGED:
            return None
        path = Path(event.path)
        name = path.name

        # Skip if this IS a test file
        if "test" in name.lower():
            return None

        # Check if there's a corresponding test file
        if path.suffix == ".py":
            test_name = f"test_{name}"
            test_path = path.parent / test_name
            tests_dir_path = path.parent.parent / "tests" / test_name

            # Check recent changes — was the test file also changed?
            recent = self._change_history[-20:]
            test_changed = any(
                test_name in Path(e.path).name
                for e in recent
                if e.event_type == EventType.FILE_CHANGED
            )

            if not test_changed and self._change_counts[event.path] >= 3:
                return Insight(
                    message=f"You've changed {name} {self._change_counts[event.path]} times but haven't updated its tests",
                    severity="warning",
                    related_files=[event.path],
                )
        return None

    def _check_large_file(self, event: WatchEvent) -> Insight | None:
        """Detect files growing too large."""
        if event.event_type not in (EventType.FILE_CHANGED, EventType.FILE_CREATED):
            return None
        try:
            size = Path(event.path).stat().st_size
            lines = Path(event.path).read_text().count("\n")
        except (OSError, UnicodeDecodeError):
            return None

        name = Path(event.path).name
        old_size = self._file_sizes.get(event.path, 0)
        self._file_sizes[event.path] = size

        if lines > 400:
            return Insight(
                message=f"{name} is {lines} lines — consider splitting it into smaller modules",
                severity="suggestion",
                related_files=[event.path],
            )
        return None

    def _check_rapid_changes(self, event: WatchEvent) -> Insight | None:
        """Detect rapid back-and-forth changes (might indicate debugging)."""
        if event.event_type != EventType.FILE_CHANGED:
            return None

        # Count changes in last 60 seconds
        now = time.time()
        recent_changes = [
            e for e in self._change_history[-50:]
            if e.path == event.path and now - e.timestamp < 60
        ]

        if len(recent_changes) >= 4:
            name = Path(event.path).name
            return Insight(
                message=f"{name} changed {len(recent_changes)} times in the last minute — need help debugging?",
                severity="info",
                related_files=[event.path],
            )
        return None

    def get_pending_insights(self) -> list[Insight]:
        """Get and clear all pending insights."""
        pending = list(self._insights)
        self._insights.clear()
        return pending

    @property
    def stats(self) -> dict:
        return {
            "total_events": len(self._change_history),
            "unique_files": len(self._change_counts),
            "hottest_files": self._change_counts.most_common(5),
        }
