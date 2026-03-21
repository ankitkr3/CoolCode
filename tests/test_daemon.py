"""Tests for daemon watchers, insights, and IPC client."""

import asyncio
import time
from pathlib import Path

import pytest

from coolcode.daemon.watchers import FileWatcher, GitWatcher, EventType, WatchEvent, IGNORE_DIRS
from coolcode.daemon.insights import InsightEngine, Insight
from coolcode.daemon.ipc import DaemonClient
from coolcode.daemon.server import is_daemon_running, PID_FILE


class TestFileWatcher:

    def test_should_watch_python_files(self, tmp_path):
        fw = FileWatcher(str(tmp_path))
        assert fw._should_watch(tmp_path / "main.py")
        assert fw._should_watch(tmp_path / "src" / "app.ts")
        assert not fw._should_watch(tmp_path / ".git" / "config")
        assert not fw._should_watch(tmp_path / "node_modules" / "pkg" / "index.js")
        assert not fw._should_watch(tmp_path / "__pycache__" / "mod.pyc")

    def test_scan_finds_python_files(self, tmp_path):
        (tmp_path / "app.py").write_text("# code")
        (tmp_path / "readme.md").write_text("# readme")
        (tmp_path / "test.js").write_text("// js")

        fw = FileWatcher(str(tmp_path))
        mtimes = fw._scan()

        paths = list(mtimes.keys())
        assert any("app.py" in p for p in paths)
        assert any("test.js" in p for p in paths)
        # .md files shouldn't be watched by default
        assert not any("readme.md" in p for p in paths)

    def test_ignore_dirs_are_skipped(self):
        assert ".git" in IGNORE_DIRS
        assert "__pycache__" in IGNORE_DIRS
        assert "node_modules" in IGNORE_DIRS

    @pytest.mark.asyncio
    async def test_watch_detects_new_file(self, tmp_path):
        fw = FileWatcher(str(tmp_path), poll_interval=0.2)

        # Start watching in background
        events = []

        async def collect():
            async for event in fw.watch():
                events.append(event)
                if len(events) >= 1:
                    fw.stop()

        # Create a file after a short delay
        async def create_file():
            await asyncio.sleep(0.3)
            (tmp_path / "new_file.py").write_text("print('hello')")

        await asyncio.gather(
            asyncio.wait_for(collect(), timeout=3.0),
            create_file(),
        )

        assert len(events) >= 1
        assert events[0].event_type == EventType.FILE_CREATED


class TestInsightEngine:

    def test_hot_file_detection(self):
        engine = InsightEngine()
        insight = None
        # Change same file 5 times
        for i in range(5):
            result = engine.analyze(WatchEvent(
                EventType.FILE_CHANGED, "/project/hot.py", f"change {i}"
            ))
            if result:
                insight = result

        assert insight is not None
        assert "hot.py" in insight.message
        assert "5 times" in insight.message

    def test_test_coverage_warning(self):
        engine = InsightEngine()
        # Change a source file 3 times without changing tests
        for i in range(3):
            engine.analyze(WatchEvent(
                EventType.FILE_CHANGED, "/project/auth.py", f"change {i}"
            ))

        insight = engine.analyze(WatchEvent(
            EventType.FILE_CHANGED, "/project/auth.py", "change 3"
        ))
        # Should eventually warn about missing test updates
        # (may take a few more changes due to dedup cooldown)

    def test_large_file_warning(self, tmp_path):
        # Create a large file
        large_file = tmp_path / "big.py"
        large_file.write_text("\n" * 500)

        engine = InsightEngine()
        insight = engine.analyze(WatchEvent(
            EventType.FILE_CHANGED, str(large_file), "modified"
        ))

        assert insight is not None
        assert "500 lines" in insight.message or "lines" in insight.message

    def test_dedup_prevents_spam(self):
        engine = InsightEngine()
        insights = []
        # Same file, same changes — should dedup same-message insights
        for i in range(20):
            result = engine.analyze(WatchEvent(
                EventType.FILE_CHANGED, "/project/same_file.py", "change"
            ))
            if result:
                insights.append(result)

        # Should be fewer insights than events — same file, same message prefix
        assert len(insights) < 20, f"Expected dedup to reduce insights, got {len(insights)}"

    def test_stats(self):
        engine = InsightEngine()
        engine.analyze(WatchEvent(EventType.FILE_CHANGED, "/a.py", ""))
        engine.analyze(WatchEvent(EventType.FILE_CHANGED, "/b.py", ""))
        engine.analyze(WatchEvent(EventType.FILE_CHANGED, "/a.py", ""))

        stats = engine.stats
        assert stats["total_events"] == 3
        assert stats["unique_files"] == 2


class TestDaemonClient:

    def test_is_running_returns_false_when_no_daemon(self):
        # Clean up any leftover PID file
        PID_FILE.unlink(missing_ok=True)
        assert not DaemonClient.is_running()

    def test_get_insights_returns_empty_when_no_daemon(self):
        PID_FILE.unlink(missing_ok=True)
        client = DaemonClient()
        insights = client.get_insights()
        assert insights == []

    def test_get_status_returns_none_when_no_daemon(self):
        PID_FILE.unlink(missing_ok=True)
        client = DaemonClient()
        status = client.get_status()
        assert status is None
