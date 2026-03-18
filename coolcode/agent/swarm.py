"""Swarm orchestration — the hive mind that coordinates queen + workers."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel

from coolcode.agent.consensus import ConsensusEngine
from coolcode.agent.queen import QueenAgent, QueenType
from coolcode.agent.router import TaskRouter
from coolcode.agent.worker import WorkerAgent, WorkerResult, WorkerType
from coolcode.config import CoolCodeConfig, SwarmConfig
from coolcode.llm.provider import LLMProvider
from coolcode.learner import WorkflowLearner
from coolcode.memory.collective import CollectiveMemory, MemoryType
from coolcode.optimizer import TokenOptimizer
from coolcode.status import StatusTracker

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
        status_tracker: StatusTracker | None = None,
        token_optimizer: TokenOptimizer | None = None,
        learner: WorkflowLearner | None = None,
    ):
        self.config = config
        self.llm_provider = llm_provider
        self.collective_memory = collective_memory
        self.task_router = task_router or TaskRouter()
        self.status = status_tracker or StatusTracker()
        self.optimizer = token_optimizer or TokenOptimizer(
            cache_dir=str(Path(config.project_dir) / ".coolcode")
        )
        self.learner = learner or WorkflowLearner(
            persist_path=str(Path.home() / ".coolcode" / "learnings.json")
        )

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
        provider_tag = provider_key.split(":")[0] if provider_key else "default"
        worker_id = f"{worker_type.value}-{provider_tag}-{index}"
        return WorkerAgent(
            worker_id=worker_id,
            worker_type=worker_type,
            model=model,
            tools=tools,
            status_tracker=self.status,
        )

    async def execute(self, task: str, tools: list | None = None) -> SwarmResult:
        """Execute a task through the swarm.

        Flow:
        0. Check cache — if identical task was solved before, return instantly
        1. Route the task to appropriate worker types
        2. Use learner to optimize worker count (skip unnecessary workers)
        3. Spawn workers across ALL providers in parallel
        4. Consensus to pick best result
        5. Cache result + record learning for next time
        """
        self.status.emit("swarm", "analyzing", "samajh rha hoon kya karna hai...")

        # Step 0: Check cache — instant result if we've seen this before
        worker_types = self.delegator.decide_worker_types(task)
        for wt in worker_types:
            cached = self.optimizer.check_cache(task, wt.value)
            if cached:
                self.status.emit("cache", "HIT", f"pehle se answer hai! ({wt.value})")
                return SwarmResult(
                    output=cached,
                    worker_results=[],
                    worker_types_used=worker_types,
                    best_worker=None,
                    consensus_algorithm="cache",
                )

        # Step 1: Route
        self.status.emit(
            "queen",
            "routing",
            f"decided workers: {', '.join(w.value for w in worker_types)}",
        )

        # Refine with learned patterns
        learned = self.task_router.suggest_workers(task)
        if learned:
            for wt, conf in learned:
                if conf > 0.7 and wt not in worker_types:
                    worker_types.insert(0, wt)
                    self.status.emit(
                        "router",
                        "learned",
                        f"adding {wt.value} (past success: {conf:.0%})",
                    )

        # Step 1.5: Learner — skip workers that always fail, optimize count
        skip_types = self.learner.suggest_skip_workers(task)
        if skip_types:
            before = len(worker_types)
            worker_types = [wt for wt in worker_types if wt.value not in skip_types]
            if len(worker_types) < before:
                self.status.emit(
                    "learner",
                    "optimized",
                    f"skipped {before - len(worker_types)} low-performing worker types",
                )
        if not worker_types:
            worker_types = [WorkerType.CODER]  # fallback

        # Learner suggests worker count based on past complexity
        suggested_count = self.learner.suggest_worker_count(task)
        if suggested_count < self._num_workers:
            self.status.emit(
                "learner",
                "optimized",
                f"using {suggested_count} workers instead of {self._num_workers} (past patterns show high confidence)",
            )

        # Step 2: Spawn workers
        from coolcode.tools.tracked import make_tracked_tools

        all_providers = self.llm_provider.get_all_models()
        workers: list[WorkerAgent] = []

        for i, wt in enumerate(worker_types):
            if self._parallel_racing and len(all_providers) > 1:
                for j, (provider_key, _model) in enumerate(all_providers):
                    provider_tag = provider_key.split(":")[0]
                    worker_id = f"{wt.value}-{provider_tag}-{i * 10 + j}"
                    tracked_tools = make_tracked_tools(self.status, worker_id)
                    workers.append(
                        self._create_worker(wt, i * 10 + j, tracked_tools, provider_key=provider_key)
                    )
            else:
                worker_id = f"{wt.value}-default-{i}"
                tracked_tools = make_tracked_tools(self.status, worker_id)
                workers.append(self._create_worker(wt, i, tracked_tools))

        provider_names = [pk.split(":")[0] for pk, _ in all_providers]
        self.status.emit(
            "swarm",
            "spawning",
            f"{len(workers)} workers across {', '.join(provider_names)}",
        )

        # Step 3: Execute ALL workers in parallel
        self.status.emit("swarm", "racing", "saare agents lage hue hain...")
        coros = [w.execute(task) for w in workers]
        results: list[WorkerResult] = await asyncio.gather(*coros)

        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        if failed:
            self.status.emit(
                "swarm", "warning", f"{len(failed)} workers failed"
            )

        # Step 4: Consensus + merge
        self.status.emit("queen", "evaluating", "sabse accha result chun rhi hoon...")

        if len(successful) <= 1:
            final_output = successful[0].output if successful else "All workers failed."
            best_result = successful[0] if successful else None
        else:
            consensus_result = await self.evaluator.evaluate_results(successful)

            if consensus_result.winner:
                best_result = next(
                    (r for r in successful if r.worker_id == consensus_result.winner),
                    successful[0],
                )
                final_output = best_result.output
                self.status.emit(
                    "queen",
                    "consensus",
                    f"winner: {best_result.worker_id} ({consensus_result.agreement_ratio:.0%} agreement)",
                )
            else:
                self.status.emit("queen", "merging", "koi clear winner nahi, merge kar rhi hoon...")
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

        # Step 5.5: Cache successful result for next time
        if best_result and best_result.success:
            self.optimizer.cache_result(task, best_result.worker_type.value, final_output)
            self.status.emit("cache", "saved", "result cached for next time")

        # Step 5.6: Record learning — what worked, what didn't
        for r in results:
            self.learner.record_execution(
                task=task,
                worker_type=r.worker_type.value,
                success=r.success,
                confidence=r.confidence,
                duration_ms=r.elapsed_ms,
            )
        self.learner.save()
        self.optimizer.save()

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

        self.status.emit("swarm", "done", "ho gya bhai!")

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
