"""Swarm orchestration — the hive mind that coordinates queen + workers."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from langchain_core.language_models import BaseChatModel

from coolcode.agent.consensus import ConsensusEngine
from coolcode.agent.queen import QueenAgent, QueenType
from coolcode.agent.router import TaskRouter
from coolcode.agent.worker import WorkerAgent, WorkerResult, WorkerType
from coolcode.config import CoolCodeConfig, SwarmConfig
from coolcode.llm.provider import LLMProvider
from coolcode.memory.collective import CollectiveMemory, MemoryType

logger = logging.getLogger(__name__)


class Swarm:
    """The hive mind: queen-led swarm with shared memory, consensus, and parallel racing.

    Flow:
    1. Task arrives → Queen Delegator routes to appropriate worker types
    2. TaskRouter refines routing based on learned patterns
    3. Multiple workers execute in parallel (racing)
    4. Queen Evaluator picks the best result via consensus
    5. Queen Coordinator merges if needed
    6. Result stored in collective memory for future learning
    """

    def __init__(
        self,
        config: CoolCodeConfig,
        llm_provider: LLMProvider,
        collective_memory: CollectiveMemory | None = None,
        task_router: TaskRouter | None = None,
    ):
        self.config = config
        self.llm_provider = llm_provider
        self.collective_memory = collective_memory
        self.task_router = task_router or TaskRouter()

        swarm_cfg = config.swarm
        self.consensus = ConsensusEngine(
            algorithm=swarm_cfg.consensus_algorithm,
            fault_tolerance=swarm_cfg.fault_tolerance_ratio,
        )

        # Initialize queens
        model = llm_provider.get_model()
        self.delegator = QueenAgent(QueenType.DELEGATOR, model=model, consensus=self.consensus)
        self.evaluator = QueenAgent(QueenType.EVALUATOR, model=model, consensus=self.consensus)
        self.coordinator = QueenAgent(QueenType.COORDINATOR, model=model, consensus=self.consensus)

        self._num_workers = swarm_cfg.num_workers
        self._parallel_racing = swarm_cfg.parallel_racing

    def _create_worker(
        self,
        worker_type: WorkerType,
        index: int,
        tools: list | None = None,
        provider_key: str | None = None,
    ) -> WorkerAgent:
        """Create a worker agent of the given type, optionally pinned to a specific provider."""
        if provider_key:
            model = self.llm_provider.get_model(provider_key)
        else:
            model = self.llm_provider.get_model()
        # Include provider name in worker_id so we can see which provider won
        provider_tag = provider_key.split(":")[0] if provider_key else "default"
        worker_id = f"{worker_type.value}-{provider_tag}-{index}"
        return WorkerAgent(
            worker_id=worker_id,
            worker_type=worker_type,
            model=model,
            tools=tools,
        )

    async def execute(self, task: str, tools: list | None = None) -> SwarmResult:
        """Execute a task through the swarm.

        1. Route the task to appropriate worker types
        2. Spawn workers across ALL providers in parallel (Claude + MiniMax race)
        3. Collect results and reach consensus
        4. Return the best/merged result
        """
        logger.info(f"Swarm executing: {task[:100]}...")

        # Step 1: Determine worker types
        worker_types = self.delegator.decide_worker_types(task)

        # Refine with learned patterns
        learned = self.task_router.suggest_workers(task)
        if learned:
            for wt, conf in learned:
                if conf > 0.7 and wt not in worker_types:
                    worker_types.insert(0, wt)

        logger.info(f"Routing to workers: {[w.value for w in worker_types]}")

        # Step 2: Spawn workers across ALL providers
        # Each worker type gets one worker per provider — Claude and MiniMax race each other
        all_providers = self.llm_provider.get_all_models()
        workers: list[WorkerAgent] = []

        for i, wt in enumerate(worker_types):
            if self._parallel_racing and len(all_providers) > 1:
                # Race the same worker type across every provider (e.g., coder-claude vs coder-minimax)
                for j, (provider_key, _model) in enumerate(all_providers):
                    workers.append(
                        self._create_worker(wt, i * 10 + j, tools, provider_key=provider_key)
                    )
            else:
                # Single provider — fall back to top-ranked
                workers.append(self._create_worker(wt, i, tools))

        provider_names = [pk.split(":")[0] for pk, _ in all_providers]
        logger.info(
            f"Spawning {len(workers)} workers across providers: {provider_names}"
        )

        # Step 3: Execute ALL workers in parallel (Claude + MiniMax racing simultaneously)
        coros = [w.execute(task) for w in workers]
        results: list[WorkerResult] = await asyncio.gather(*coros)

        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        if failed:
            logger.warning(f"{len(failed)} workers failed: {[r.worker_id for r in failed]}")

        # Step 4: Consensus + merge
        if len(successful) <= 1:
            final_output = successful[0].output if successful else "All workers failed."
            best_result = successful[0] if successful else None
        else:
            consensus_result = await self.evaluator.evaluate_results(successful)

            if consensus_result.winner:
                # Find the winning result
                best_result = next(
                    (r for r in successful if r.worker_id == consensus_result.winner),
                    successful[0],
                )
                final_output = best_result.output
            else:
                # No consensus — merge results
                final_output = await self.coordinator.merge_results(task, successful)
                best_result = max(successful, key=lambda r: r.confidence)

        # Step 5: Learn from this execution
        quality = best_result.confidence if best_result else 0.0
        self.task_router.record_outcome(
            task=task,
            routed_to=worker_types,
            success=bool(successful),
            quality_score=quality,
        )

        # Step 6: Store in collective memory
        if self.collective_memory and best_result:
            self.collective_memory.store(
                memory_id=f"task-{hash(task) % 100000}",
                memory_type=MemoryType.INSIGHT,
                content=f"Task: {task[:200]}\nRouted to: {[w.value for w in worker_types]}\n"
                f"Best worker: {best_result.worker_id} (confidence: {best_result.confidence:.2f})",
                tags=[w.value for w in worker_types],
                relevance_score=quality,
            )

        return SwarmResult(
            output=final_output,
            worker_results=results,
            worker_types_used=worker_types,
            best_worker=best_result,
            consensus_algorithm=self.consensus.algorithm,
        )


class SwarmResult:
    """The final output from a swarm execution."""

    def __init__(
        self,
        output: str,
        worker_results: list[WorkerResult],
        worker_types_used: list[WorkerType],
        best_worker: WorkerResult | None = None,
        consensus_algorithm: str = "",
    ):
        self.output = output
        self.worker_results = worker_results
        self.worker_types_used = worker_types_used
        self.best_worker = best_worker
        self.consensus_algorithm = consensus_algorithm

    @property
    def stats(self) -> dict[str, Any]:
        successful = [r for r in self.worker_results if r.success]

        # Count wins per provider from worker IDs like "coder-claude-0", "coder-minimax-1"
        provider_results: dict[str, dict[str, int]] = {}
        for r in self.worker_results:
            parts = r.worker_id.split("-")
            provider = parts[1] if len(parts) >= 3 else "unknown"
            if provider not in provider_results:
                provider_results[provider] = {"total": 0, "success": 0}
            provider_results[provider]["total"] += 1
            if r.success:
                provider_results[provider]["success"] += 1

        winning_provider = "N/A"
        if self.best_worker:
            parts = self.best_worker.worker_id.split("-")
            winning_provider = parts[1] if len(parts) >= 3 else "unknown"

        return {
            "total_workers": len(self.worker_results),
            "successful": len(successful),
            "failed": len(self.worker_results) - len(successful),
            "worker_types": [w.value for w in self.worker_types_used],
            "best_worker": self.best_worker.worker_id if self.best_worker else None,
            "best_confidence": self.best_worker.confidence if self.best_worker else 0.0,
            "winning_provider": winning_provider,
            "provider_breakdown": provider_results,
            "consensus": self.consensus_algorithm,
            "avg_latency_ms": (
                sum(r.elapsed_ms for r in successful) / len(successful) if successful else 0
            ),
        }
