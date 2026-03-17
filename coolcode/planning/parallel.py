"""Parallel agent racing — spawn multiple agents, best one wins."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from coolcode.agent.consensus import ConsensusEngine, Vote
from coolcode.agent.worker import WorkerAgent, WorkerResult, WorkerType

logger = logging.getLogger(__name__)


@dataclass
class RaceResult:
    """Result of a parallel race between agents."""

    winner: WorkerResult
    all_results: list[WorkerResult]
    selection_method: str  # "fastest", "consensus", "highest_confidence"


class ParallelRacer:
    """Race multiple agents on the same task, pick the best result.

    Three selection strategies:
    - "fastest": first successful result wins (cancel others)
    - "consensus": wait for all, use consensus to pick winner
    - "highest_confidence": wait for all, pick highest self-rated confidence
    """

    def __init__(
        self,
        strategy: str = "highest_confidence",
        consensus: ConsensusEngine | None = None,
    ):
        self.strategy = strategy
        self.consensus = consensus or ConsensusEngine(algorithm="weighted")

    async def race(
        self,
        task: str,
        workers: list[WorkerAgent],
    ) -> RaceResult:
        """Race workers on the same task."""
        if not workers:
            raise ValueError("No workers to race")

        if self.strategy == "fastest":
            return await self._race_fastest(task, workers)
        else:
            return await self._race_all(task, workers)

    async def _race_fastest(self, task: str, workers: list[WorkerAgent]) -> RaceResult:
        """First successful result wins."""
        tasks_list = [w.execute(task) for w in workers]

        # Use asyncio to get the first completed successful result
        done: list[WorkerResult] = []
        pending = set()

        for coro in tasks_list:
            pending.add(asyncio.ensure_future(coro))

        winner: WorkerResult | None = None
        while pending:
            finished, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for fut in finished:
                result = fut.result()
                done.append(result)
                if result.success and winner is None:
                    winner = result
                    # Cancel remaining
                    for p in pending:
                        p.cancel()
                    # Collect cancelled results
                    for p in pending:
                        try:
                            await p
                        except (asyncio.CancelledError, Exception):
                            pass
                    pending = set()
                    break

        if winner is None:
            # All failed — return the one with highest confidence anyway
            winner = max(done, key=lambda r: r.confidence) if done else done[0]

        return RaceResult(winner=winner, all_results=done, selection_method="fastest")

    async def _race_all(self, task: str, workers: list[WorkerAgent]) -> RaceResult:
        """Wait for all workers, then pick the best."""
        results: list[WorkerResult] = await asyncio.gather(
            *[w.execute(task) for w in workers]
        )

        successful = [r for r in results if r.success]
        if not successful:
            best = max(results, key=lambda r: r.confidence) if results else results[0]
            return RaceResult(winner=best, all_results=results, selection_method="fallback")

        if self.strategy == "consensus":
            votes = [
                Vote(agent_id=r.worker_id, choice=r.worker_id, confidence=r.confidence)
                for r in successful
            ]
            result = self.consensus.resolve(votes)
            if result.winner:
                winner = next(r for r in successful if r.worker_id == result.winner)
            else:
                winner = max(successful, key=lambda r: r.confidence)
            return RaceResult(winner=winner, all_results=results, selection_method="consensus")

        else:  # highest_confidence
            winner = max(successful, key=lambda r: r.confidence)
            return RaceResult(
                winner=winner, all_results=results, selection_method="highest_confidence"
            )
