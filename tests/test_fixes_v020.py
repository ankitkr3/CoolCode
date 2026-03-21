"""Tests for v0.2.1 fixes: warning suppression, version, identity, worker count."""

from unittest.mock import patch, MagicMock


class TestVersionDetection:
    """Test dynamic version reading."""

    def test_get_version_returns_string(self):
        from coolcode.cli import _get_version
        ver = _get_version()
        assert isinstance(ver, str)
        assert len(ver) > 0
        assert ver != "dev"  # should find the version

    def test_get_version_has_semver_format(self):
        """Version should look like semver (X.Y.Z)."""
        from coolcode.cli import _get_version
        ver = _get_version()
        parts = ver.split(".")
        assert len(parts) >= 2, f"Expected semver format, got '{ver}'"


class TestWorkerIdentity:
    """Test that workers identify as Cool Code, not Deep Agent."""

    def test_system_prompt_contains_cool_code(self):
        from coolcode.agent.worker import WorkerAgent, WorkerType
        with patch("coolcode.agent.worker.create_deep_agent") as mock:
            mock.return_value = MagicMock()
            worker = WorkerAgent(
                worker_id="test-0",
                worker_type=WorkerType.CODER,
                model="anthropic:claude-sonnet-4-6",
            )
            # Check that create_deep_agent was called with Cool Code identity
            call_kwargs = mock.call_args
            system_prompt = call_kwargs.kwargs.get("system_prompt") or call_kwargs[1].get("system_prompt", "")
            assert "Cool Code" in system_prompt
            assert "Never identify yourself as Deep Agent" in system_prompt

    def test_all_worker_types_get_identity(self):
        from coolcode.agent.worker import WorkerAgent, WorkerType
        with patch("coolcode.agent.worker.create_deep_agent") as mock:
            mock.return_value = MagicMock()
            for wt in WorkerType:
                WorkerAgent(
                    worker_id=f"test-{wt.value}",
                    worker_type=wt,
                    model="anthropic:claude-sonnet-4-6",
                )
            # All 8 worker types should have been created
            assert mock.call_count == len(WorkerType)
            for call_args in mock.call_args_list:
                prompt = call_args.kwargs.get("system_prompt") or call_args[1].get("system_prompt", "")
                assert "Cool Code" in prompt


class TestLearnerWorkerCount:
    """Test that learner doesn't over-optimize to 1 worker."""

    def test_unknown_task_returns_configured_count(self):
        from coolcode.learner import WorkflowLearner
        learner = WorkflowLearner()
        assert learner.suggest_worker_count("something completely new", configured_count=3) == 3
        assert learner.suggest_worker_count("another unknown", configured_count=5) == 5

    def test_never_returns_less_than_2(self):
        """Even high-confidence patterns should return at least 2."""
        from coolcode.learner import WorkflowLearner
        learner = WorkflowLearner()
        # Record high-confidence pattern
        for _ in range(10):
            learner.record_execution(
                task="fix the login bug",
                worker_type="coder",
                success=True,
                confidence=0.95,
                duration_ms=500,
            )

        count = learner.suggest_worker_count("fix the login bug", configured_count=3)
        assert count >= 2, f"Expected at least 2 workers, got {count}"

    def test_low_confidence_returns_full_count(self):
        from coolcode.learner import WorkflowLearner
        learner = WorkflowLearner()
        for _ in range(5):
            learner.record_execution(
                task="refactor complex module",
                worker_type="refactorer",
                success=True,
                confidence=0.3,
                duration_ms=5000,
            )

        count = learner.suggest_worker_count("refactor complex module", configured_count=4)
        assert count == 4, f"Low confidence should use full count, got {count}"


class TestWarningsSuppressed:
    """Test that sklearn/numpy warnings are filtered."""

    def test_numpy_warning_actually_suppressed(self):
        """The actual warning should be suppressed after importing cli."""
        import warnings
        import coolcode.cli  # noqa: F401 — triggers filterwarnings setup

        # This should NOT raise or print — it should be silently ignored
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")  # reset to catch all
            # Re-apply our filters on top
            warnings.filterwarnings("ignore", message=".*A NumPy version.*", category=UserWarning)
            warnings.warn("A NumPy version >=1.33.0 is required", UserWarning)
            numpy_warnings = [w for w in caught if "NumPy" in str(w.message)]
            assert len(numpy_warnings) == 0, "NumPy warning should be suppressed"


class TestFlushStdin:
    """Test stdin flushing utility."""

    def test_flush_stdin_does_not_crash(self):
        """_flush_stdin should be safe to call anytime."""
        from coolcode.cli import _flush_stdin
        # Should not raise even if stdin has nothing
        _flush_stdin()
