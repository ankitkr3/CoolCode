"""Tests to verify goal pipelines don't use provider racing."""

from unittest.mock import MagicMock, patch, call

from coolcode.tools.tracked import FileReadCache, make_tracked_tools
from coolcode.status import StatusTracker


class TestGoalNoRacing:
    """Verify that goal pipeline workers use a single provider (no racing)."""

    def test_goal_worker_uses_single_model(self):
        """_create_goal_worker should call get_model() once (not get_all_models)."""
        from coolcode.agent.swarm import Swarm
        from coolcode.agent.worker import WorkerType

        with patch.object(Swarm, "__init__", lambda self, **kwargs: None):
            s = Swarm.__new__(Swarm)
            s.llm_provider = MagicMock()
            s.llm_provider.get_model.return_value = "anthropic:claude-sonnet-4-6"
            s.status = StatusTracker()

            with patch("coolcode.agent.worker.create_deep_agent") as mock_agent:
                mock_agent.return_value = MagicMock()
                worker = s._create_goal_worker(
                    worker_type=WorkerType.SECURITY,
                    index=0,
                    prompt_override="test prompt",
                    tools=[],
                    timeout=60,
                )

                # Should call get_model (single) not get_all_models
                s.llm_provider.get_model.assert_called_once()
                s.llm_provider.get_all_models.assert_not_called()

    def test_tracked_tools_with_file_cache(self):
        """make_tracked_tools should accept and use a shared FileReadCache."""
        cache = FileReadCache()
        status = StatusTracker()

        tools1 = make_tracked_tools(status, "worker-1", file_cache=cache)
        tools2 = make_tracked_tools(status, "worker-2", file_cache=cache)

        # Both should return tool lists
        assert len(tools1) > 0
        assert len(tools2) > 0

        # Find the read_file tool
        read_fn1 = next(t for t in tools1 if t.__name__ == "read_file")
        read_fn2 = next(t for t in tools2 if t.__name__ == "read_file")

        # Both should be callable
        assert callable(read_fn1)
        assert callable(read_fn2)

    def test_file_cache_shared_across_workers(self, tmp_path):
        """Two workers sharing a cache should not re-read the same file."""
        # Create a test file
        test_file = tmp_path / "test.py"
        test_file.write_text("line 1\nline 2\nline 3\n")

        cache = FileReadCache()
        status = StatusTracker()

        tools1 = make_tracked_tools(status, "worker-1", file_cache=cache)
        tools2 = make_tracked_tools(status, "worker-2", file_cache=cache)

        read_fn1 = next(t for t in tools1 if t.__name__ == "read_file")
        read_fn2 = next(t for t in tools2 if t.__name__ == "read_file")

        # Worker 1 reads the file — cache miss
        result1 = read_fn1(str(test_file))
        assert "line 1" in result1
        assert cache.misses == 1
        assert cache.hits == 0

        # Worker 2 reads the same file — cache hit
        result2 = read_fn2(str(test_file))
        assert result2 == result1  # exact same content
        assert cache.hits == 1
        assert cache.misses == 1  # didn't increase

    def test_no_cache_means_no_caching(self, tmp_path):
        """When file_cache is None, reads go through normally without caching."""
        test_file = tmp_path / "test.py"
        test_file.write_text("hello\n")

        status = StatusTracker()
        tools = make_tracked_tools(status, "worker-1", file_cache=None)

        read_fn = next(t for t in tools if t.__name__ == "read_file")
        result = read_fn(str(test_file))
        assert "hello" in result
        # No crash — just works without cache
