"""Tests for ESC cancellation propagation in swarm execution."""

import asyncio
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from coolcode.agent.worker import WorkerResult, WorkerType


@dataclass
class FakeSwarmForCancel:
    """Minimal swarm-like object to test _gather_with_cancel in isolation."""

    _cancelled: bool = False

    def cancel(self):
        self._cancelled = True


class TestGatherWithCancel:
    """Tests for Swarm._gather_with_cancel — the cancellation-aware gather."""

    @pytest.fixture
    def swarm(self):
        """Create a minimal Swarm-like object with _gather_with_cancel."""
        from coolcode.agent.swarm import Swarm
        from coolcode.config import CoolCodeConfig, LLMConfig
        from coolcode.status import StatusTracker

        # We need a real Swarm but with mocked LLM provider
        config = CoolCodeConfig(
            project_dir="/tmp/test",
            providers=[LLMConfig(provider="claude", model="test", api_key="fake")],
        )
        with patch("coolcode.agent.swarm.LLMProvider") as MockProvider:
            mock_provider = MagicMock()
            mock_provider.get_model.return_value = MagicMock()
            mock_provider.get_all_models.return_value = []
            # Patch the Swarm __init__ to avoid LLM setup
            with patch.object(Swarm, "__init__", lambda self, **kwargs: None):
                s = Swarm.__new__(Swarm)
                s._cancelled = False
                s._injected_context = []
                s.status = StatusTracker()
                return s

    @pytest.mark.asyncio
    async def test_all_complete_normally(self, swarm):
        """When no cancellation, all coroutines complete and return results."""
        async def fast_worker(idx):
            await asyncio.sleep(0.05)
            return WorkerResult(
                worker_id=f"test-{idx}",
                worker_type=WorkerType.CODER,
                output=f"result-{idx}",
                confidence=0.9,
                elapsed_ms=50,
            )

        coros = [fast_worker(i) for i in range(3)]
        results = await swarm._gather_with_cancel(coros, poll_interval=0.1)

        assert len(results) == 3
        assert all(r.output.startswith("result-") for r in results)
        assert all(r.error is None for r in results)

    @pytest.mark.asyncio
    async def test_cancel_stops_pending_workers(self, swarm):
        """When cancelled, pending workers get cancelled results."""
        async def slow_worker(idx):
            await asyncio.sleep(10)  # very slow — should get cancelled
            return WorkerResult(
                worker_id=f"test-{idx}",
                worker_type=WorkerType.CODER,
                output=f"result-{idx}",
                confidence=0.9,
                elapsed_ms=10000,
            )

        coros = [slow_worker(i) for i in range(3)]

        # Cancel after a short delay
        async def cancel_after():
            await asyncio.sleep(0.2)
            swarm._cancelled = True

        cancel_task = asyncio.create_task(cancel_after())
        results = await swarm._gather_with_cancel(coros, poll_interval=0.1)
        await cancel_task

        # All should have error (cancelled)
        assert len(results) == 3
        cancelled = [r for r in results if r.error and "Cancel" in r.error]
        assert len(cancelled) == 3, f"Expected 3 cancelled, got errors: {[r.error for r in results]}"

    @pytest.mark.asyncio
    async def test_mix_of_fast_and_cancelled(self, swarm):
        """Fast workers complete before cancel, slow ones get cancelled."""
        async def fast_worker():
            await asyncio.sleep(0.05)
            return WorkerResult(
                worker_id="fast",
                worker_type=WorkerType.CODER,
                output="done",
                confidence=0.9,
                elapsed_ms=50,
            )

        async def slow_worker():
            await asyncio.sleep(10)
            return WorkerResult(
                worker_id="slow",
                worker_type=WorkerType.CODER,
                output="never",
                confidence=0.9,
                elapsed_ms=10000,
            )

        coros = [fast_worker(), slow_worker()]

        # Cancel after fast completes but before slow
        async def cancel_after():
            await asyncio.sleep(0.3)
            swarm._cancelled = True

        cancel_task = asyncio.create_task(cancel_after())
        results = await swarm._gather_with_cancel(coros, poll_interval=0.1)
        await cancel_task

        assert len(results) == 2
        # One should have completed, one cancelled
        completed = [r for r in results if r.error is None]
        cancelled = [r for r in results if r.error and "Cancel" in r.error]
        assert len(completed) == 1
        assert completed[0].output == "done"
        assert len(cancelled) == 1
