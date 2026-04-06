"""Swarm orchestration — the hive mind that coordinates queen + workers."""

from __future__ import annotations

import asyncio
import json
import time
import logging
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel

from coolcode.agent.consensus import ConsensusEngine
from coolcode.agent.queen import QueenAgent, QueenType
from coolcode.agent.router import TaskRouter
from coolcode.agent.worker import WorkerAgent, WorkerResult, WorkerType
from coolcode.config import CoolCodeConfig, SwarmConfig
from coolcode.cost import CostTracker
from coolcode.llm.provider import LLMProvider
from coolcode.learner import WorkflowLearner
from coolcode.memory.collective import CollectiveMemory, MemoryType
from coolcode.memory.knowledge_graph import KnowledgeGraph
from coolcode.memory.learning_bridge import LearningBridge
from coolcode.memory.scoped import ScopedMemory
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
        learner: WorkflowLearner | None = None,
        knowledge_graph: KnowledgeGraph | None = None,
        scoped_memory: ScopedMemory | None = None,
        cost_tracker: CostTracker | None = None,
    ):
        self.config = config
        self.llm_provider = llm_provider
        self.collective_memory = collective_memory
        self.task_router = task_router or TaskRouter()
        self.status = status_tracker or StatusTracker()
        self.cost_tracker = cost_tracker or CostTracker()
        self.learner = learner or WorkflowLearner(
            persist_path=str(Path.home() / ".coolcode" / "learnings.json")
        )

        # Memory systems — global-by-default, fall back to project dir if opted out.
        project_dir = config.project_dir
        memory_dir = Path(config.memory.memory_dir or (Path.home() / ".coolcode"))
        memory_dir.mkdir(parents=True, exist_ok=True)
        self._memory_dir = memory_dir

        self.knowledge_graph = knowledge_graph or KnowledgeGraph(
            persist_path=str(memory_dir / "knowledge_graph.json")
        )
        self.scoped_memory = scoped_memory or ScopedMemory(project_dir)
        self.learning_bridge = LearningBridge(
            project_dir=project_dir,
            collective_memory=self.collective_memory,
            knowledge_graph=self.knowledge_graph,
            coolcode_dir=memory_dir,
        )

        # Conversation history — persisted across sessions. Primary store is
        # ``memory_dir`` (global by default). We also merge any legacy
        # per-project history file so users who predate global memory don't
        # lose context the first time they launch with the new default.
        self._history_path = memory_dir / "conversation_history.json"
        self._legacy_history_path = Path(project_dir) / ".coolcode" / "conversation_history.json"
        self._conversation_history: list[dict[str, str]] = self._load_history()

        # In-process controls — set by CLI during execution
        self._cancelled = False  # ESC pressed → stop gracefully
        self._cancel_event = asyncio.Event()  # async cancel signal for workers
        self._injected_context: list[str] = []  # user-typed info mid-execution
        self._history_lock = asyncio.Lock()  # protects _conversation_history mutations

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
        self._goal = "general"  # default — queen routes everything

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

    def set_goal(self, name: str) -> None:
        """Set the active goal. 'general' uses default queen routing."""
        self._goal = name

    def cancel(self) -> None:
        """Cancel the current execution gracefully."""
        self._cancelled = True
        self._cancel_event.set()
        self.status.emit("swarm", "cancelling", "ruko, band kar rha hoon...")

    def inject_context(self, text: str) -> None:
        """Inject additional context from the user mid-execution."""
        self._injected_context.append(text)
        self.status.emit("user", "info added", text[:80])

    def _load_history(self) -> list[dict[str, str]]:
        """Load conversation history, merging global + legacy per-project files.

        Entries are deduped by ``(role, content, ts)`` and sorted by ``ts``.
        We merge so users upgrading from per-project memory keep everything
        they had, and the next save consolidates it all into the global file.
        """
        merged: list[dict[str, str]] = []
        seen: set[tuple] = set()

        def _read(path: Path) -> list[dict[str, str]]:
            try:
                if path.exists():
                    data = json.loads(path.read_text())
                    if isinstance(data, list):
                        return data
            except (json.JSONDecodeError, IOError, KeyError) as e:
                logger.debug(f"Could not read history from {path}: {e}")
            return []

        sources = [self._history_path]
        if self._legacy_history_path != self._history_path:
            sources.append(self._legacy_history_path)

        for src in sources:
            for entry in _read(src):
                if not isinstance(entry, dict):
                    continue
                key = (entry.get("role", ""), entry.get("content", "")[:200], entry.get("ts", 0))
                if key in seen:
                    continue
                seen.add(key)
                merged.append(entry)

        merged.sort(key=lambda e: e.get("ts", 0))
        # Keep last 100 exchanges to prevent unbounded growth
        return merged[-100:]

    def _save_history(self) -> None:
        """Persist conversation history to disk (global store)."""
        try:
            self._history_path.parent.mkdir(parents=True, exist_ok=True)
            # Keep last 100 exchanges
            data = self._conversation_history[-100:]
            self._history_path.write_text(json.dumps(data, indent=2))
        except IOError as e:
            logger.warning(f"Failed to save conversation history: {e}")

    def _reset_execution_state(self) -> None:
        """Reset per-execution state before a new run."""
        self._cancelled = False
        self._cancel_event = asyncio.Event()  # fresh event per execution
        self._injected_context.clear()

    def _record_costs(self, results: list[WorkerResult], task: str = "") -> float:
        """Record costs from worker results. Returns total cost."""
        total = 0.0
        for r in results:
            input_tokens = r.metadata.get("input_tokens", 0)
            output_tokens = r.metadata.get("output_tokens", 0)
            if input_tokens or output_tokens:
                # Extract provider and model from worker_id (format: type-provider-idx)
                parts = r.worker_id.split("-")
                provider = parts[1] if len(parts) >= 3 else "unknown"
                # Find the model from the provider config
                model = ""
                for p in self.config.providers:
                    if p.provider == provider or provider == "goal" or provider == "default":
                        model = p.model
                        break
                if not model and self.config.providers:
                    model = self.config.providers[0].model
                cost = self.cost_tracker.record(
                    provider=provider,
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    task_snippet=task[:100],
                )
                total += cost
                self.status.emit(
                    r.worker_id, "cost",
                    f"${cost:.4f} ({input_tokens} in / {output_tokens} out)"
                )
        return total

    async def _gather_with_cancel(
        self, coros: list, poll_interval: float = 0.2
    ) -> list[WorkerResult]:
        """Run coroutines concurrently; stop all if ESC is pressed.

        Workers now cooperate via self._cancel_event — they check it between
        astream chunks and return partial results cleanly. This wrapper also
        hard-cancels any task that doesn't respond within a grace period.
        """
        tasks = [asyncio.ensure_future(c) for c in coros]

        while True:
            if self._cancelled:
                # Workers will see _cancel_event and return partial results naturally.
                # Give them a grace period, then hard-cancel anything still running.
                self.status.emit("goal", "cancelling", "ruko, workers band kar rha hoon...")
                try:
                    await asyncio.wait(tasks, timeout=3.0)
                except Exception:
                    pass
                for t in tasks:
                    if not t.done():
                        t.cancel()
                break

            if all(t.done() for t in tasks):
                break

            await asyncio.sleep(poll_interval)

        # Collect results — partials preserved, hard-cancels become error results
        final: list[WorkerResult] = []
        for i, t in enumerate(tasks):
            try:
                if t.done() and not t.cancelled():
                    final.append(t.result())
                else:
                    final.append(WorkerResult(
                        worker_id=f"cancelled-{i}",
                        worker_type=WorkerType.CODER,
                        output="",
                        confidence=0.0,
                        elapsed_ms=0.0,
                        error="Cancelled by user (ESC)",
                    ))
            except (asyncio.CancelledError, Exception) as e:
                final.append(WorkerResult(
                    worker_id=f"error-{i}",
                    worker_type=WorkerType.CODER,
                    output="",
                    confidence=0.0,
                    elapsed_ms=0.0,
                    error=str(e),
                ))

        return final

    @property
    def goal(self) -> str:
        return self._goal

    def _create_goal_worker(
        self,
        worker_type: WorkerType,
        index: int,
        prompt_override: str,
        tools: list | None = None,
        timeout: int = 0,
    ) -> WorkerAgent:
        """Create a worker with a goal-specific system prompt override."""
        model = self.llm_provider.get_model()
        worker_id = f"{worker_type.value}-goal-{index}"
        return WorkerAgent(
            worker_id=worker_id,
            worker_type=worker_type,
            model=model,
            tools=tools,
            status_tracker=self.status,
            system_prompt_override=prompt_override,
        )

    async def _execute_goal_pipeline(
        self, task: str, enriched_task: str, tools: list | None = None
    ) -> SwarmResult:
        """Execute a goal pipeline — predetermined worker sequence, no queen routing.

        Key optimizations vs general mode:
        - Shared file-read cache across all workers (prevents duplicate reads)
        - No parallel provider racing (goal stages already parallelize workers)
        - Cancellation propagates into running workers immediately on ESC
        """
        from coolcode.agent.goals import get_goal
        from coolcode.tools.tracked import FileReadCache, make_tracked_tools

        pipeline = get_goal(self._goal)
        if not pipeline:
            raise ValueError(f"Unknown goal: {self._goal}")

        self.status.emit("goal", "started", f"goal: {pipeline.name} — {pipeline.description}")

        # Shared file-read cache — all workers in this pipeline share it
        file_cache = FileReadCache()

        all_stage_outputs: list[str] = []
        all_results: list[WorkerResult] = []
        all_worker_types: list[WorkerType] = []

        for stage_idx, stage in enumerate(pipeline.stages):
            # Check cancellation between stages
            if self._cancelled:
                self.status.emit("goal", "cancelled", "user ne rok diya — partial results returned")
                break

            stage_labels = ", ".join(s.label for s in stage.steps)
            self.status.emit(
                "goal",
                f"stage {stage_idx + 1}/{len(pipeline.stages)}",
                stage_labels,
            )

            # Build context: enriched_task + all previous stage outputs + injected context
            stage_context = enriched_task
            for i, output in enumerate(all_stage_outputs):
                stage_context += f"\n\n--- STAGE {i + 1} RESULTS ---\n{output[:4000]}\n--- END STAGE {i + 1} ---"

            # Append any user-injected context (added mid-execution via input)
            if self._injected_context:
                stage_context += "\n\n--- ADDITIONAL USER CONTEXT ---\n"
                stage_context += "\n".join(self._injected_context)
                stage_context += "\n--- END ADDITIONAL CONTEXT ---"

            # Create workers for all steps in this stage (they run in parallel)
            # NOTE: No provider racing — goal pipelines already parallelize via multiple steps
            workers: list[WorkerAgent] = []
            for step_idx, step in enumerate(stage.steps):
                worker_idx = stage_idx * 10 + step_idx
                worker_id = f"{step.worker_type.value}-goal-{worker_idx}"
                tracked_tools = make_tracked_tools(self.status, worker_id, file_cache=file_cache)
                w = self._create_goal_worker(
                    worker_type=step.worker_type,
                    index=worker_idx,
                    prompt_override=step.prompt_override,
                    tools=tracked_tools,
                    timeout=step.timeout,
                )
                workers.append(w)
                all_worker_types.append(step.worker_type)

            # Execute all steps in this stage — with cancellation support
            coros = [
                w.execute(stage_context, timeout=step.timeout, cancel_event=self._cancel_event)
                for w, step in zip(workers, stage.steps)
            ]
            stage_results = await self._gather_with_cancel(coros)
            self._record_costs(stage_results, task)
            all_results.extend(stage_results)

            # Merge stage results
            successful = [r for r in stage_results if r.success]
            if not successful:
                self.status.emit("goal", "warning", f"stage {stage_idx + 1} failed — all workers errored")
                all_stage_outputs.append(f"[Stage {stage_idx + 1} failed: no successful workers]")
                continue

            if stage.merge_strategy == "concatenate":
                merged = "\n\n---\n\n".join(r.output for r in successful)
            else:
                # Default: use best result
                merged = max(successful, key=lambda r: r.confidence).output

            all_stage_outputs.append(merged)

            for r in successful:
                self.status.emit(
                    r.worker_id, "done",
                    f"confidence: {r.confidence:.0%}, {r.elapsed_ms:.0f}ms"
                )

        # Combine all stage outputs into final result
        final_output = "\n\n".join(all_stage_outputs)

        # Best result across all stages
        successful_all = [r for r in all_results if r.success]
        best_result = max(successful_all, key=lambda r: r.confidence) if successful_all else None

        # Learn from execution
        quality = best_result.confidence if best_result else 0.0
        for r in all_results:
            self.learner.record_execution(
                task=task,
                worker_type=r.worker_type.value,
                success=r.success,
                confidence=r.confidence,
                duration_ms=r.elapsed_ms,
                error=r.error,
            )
        self.learner.save()

        # Store in memory
        if self.collective_memory and best_result:
            self.collective_memory.store(
                memory_id=f"goal-{self._goal}-{hash(task) % 100000}",
                memory_type=MemoryType.INSIGHT,
                content=f"Goal: {self._goal}\nTask: {task[:200]}\n"
                f"Stages: {len(pipeline.stages)}, Workers: {len(all_results)}\n"
                f"Best: {best_result.worker_id} ({best_result.confidence:.2f})",
                tags=[self._goal] + [w.value for w in all_worker_types],
                relevance_score=quality,
            )

        # Conversation history (persisted) — guarded so concurrent executions don't
        # interleave appends or race on the JSON save.
        async with self._history_lock:
            self._conversation_history.append({
                "role": "assistant",
                "content": final_output[:1000],
                "ts": time.time(),
                "project": self.config.project_dir,
            })
            self._save_history()

        # Learning bridge
        if best_result:
            self.learning_bridge.learn_from_task(
                task=task,
                result=final_output[:500],
                worker_type=best_result.worker_type.value,
                confidence=best_result.confidence,
                success=best_result.success,
            )
            self.learning_bridge.save()

        # Log file cache stats
        if file_cache.hits > 0:
            self.status.emit(
                "goal", "cache",
                f"file reads saved: {file_cache.hits} cached / {file_cache.hits + file_cache.misses} total"
            )

        self.status.emit("goal", "done", f"{pipeline.name} complete — ho gya bhai!")

        return SwarmResult(
            output=final_output,
            worker_results=all_results,
            worker_types_used=list(set(all_worker_types)),
            best_worker=best_result,
            consensus_algorithm=f"goal:{pipeline.name}",
        )

    def _build_context(self, task: str) -> str:
        """Build rich context from conversation history, memory, and project info."""
        parts: list[str] = []

        # Project directory context
        parts.append(f"[Project Directory: {self.config.project_dir}]")
        parts.append(f"[User Home: {Path.home()}]")

        # YOUR CONVERSATION HISTORY (current project first, then recent global)
        if self._conversation_history:
            cur_project = self.config.project_dir
            project_entries = [
                e for e in self._conversation_history
                if e.get("project", cur_project) == cur_project
            ][-10:]
            global_entries = [
                e for e in self._conversation_history
                if e.get("project") and e.get("project") != cur_project
            ][-5:]

            if project_entries:
                parts.append("\n--- CONVERSATION HISTORY (this project) ---")
                for entry in project_entries:
                    role = entry.get("role", "?").upper()
                    parts.append(f"{role}: {entry.get('content', '')[:500]}")
                parts.append("--- END ---\n")

            if global_entries:
                parts.append("\n--- RECENT CONVERSATION HISTORY (other projects) ---")
                for entry in global_entries:
                    role = entry.get("role", "?").upper()
                    proj = Path(entry.get("project", "")).name or "?"
                    parts.append(f"[{proj}] {role}: {entry.get('content', '')[:300]}")
                parts.append("--- END ---\n")

        # YOUR MEMORY: similar tasks you've handled before
        similar = self.learning_bridge.semantic_recall(task, k=3)
        if similar:
            parts.append("\n--- YOUR MEMORY: PAST TASKS YOU'VE HANDLED ---")
            for s in similar:
                meta = s.get("metadata", {})
                parts.append(
                    f"• [{meta.get('worker_type', '?')}] (confidence: {meta.get('confidence', 0):.0%}) "
                    f"{meta.get('task', '')[:150]}"
                )
            parts.append("--- END PAST TASKS ---\n")

        # YOUR MEMORY: insights and learnings from past work
        if self.collective_memory:
            related = self.collective_memory.search(
                memory_type=MemoryType.INSIGHT, limit=5, min_relevance=0.3
            )
            if related:
                parts.append("\n--- YOUR MEMORY: INSIGHTS FROM PAST WORK ---")
                for mem in related[:3]:
                    parts.append(f"• {mem['content'][:200]}")
                parts.append("--- END INSIGHTS ---\n")

        # YOUR MEMORY: key concepts and entities you've learned about
        top_entities = self.knowledge_graph.get_pagerank(top_k=5)
        if top_entities:
            parts.append("\n--- YOUR MEMORY: KEY CONCEPTS YOU KNOW ---")
            for entity_id, score in top_entities:
                parts.append(f"• {entity_id} (relevance: {score:.3f})")
            parts.append("--- END CONCEPTS ---\n")

        # YOUR MEMORY: user preferences
        user_prefs = self.scoped_memory.recall("user", "preferences")
        if user_prefs:
            parts.append(f"\n[Your Memory — User Preferences: {json.dumps(user_prefs)[:300]}]")

        return "\n".join(parts)

    async def execute(self, task: str, tools: list | None = None) -> SwarmResult:
        """Execute a task through the swarm.

        Flow:
        1. Route the task to appropriate worker types
        2. Use learner to optimize worker count (skip unnecessary workers)
        3. Spawn workers across ALL providers in parallel
        4. Consensus to pick best result
        5. Record learning for next time
        """
        self._reset_execution_state()
        self.status.emit("swarm", "analyzing", "samajh rha hoon kya karna hai...")

        # Add user message to conversation history (persisted)
        async with self._history_lock:
            self._conversation_history.append({
                "role": "user",
                "content": task,
                "ts": time.time(),
                "project": self.config.project_dir,
            })
            self._save_history()

        # Build context from memory + history
        context = self._build_context(task)
        enriched_task = f"{context}\n\n[CURRENT TASK]\n{task}"

        # Goal pipeline — if a goal is active, bypass queen routing entirely
        if self._goal != "general":
            from coolcode.agent.goals import get_goal
            pipeline = get_goal(self._goal)
            if pipeline:
                return await self._execute_goal_pipeline(task, enriched_task, tools)

        # Step 1: Route (general mode — queen decides)
        worker_types = self.delegator.decide_worker_types(task)
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
        suggested_count = self.learner.suggest_worker_count(task, configured_count=self._num_workers)
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

        # Check cancellation before spawning
        if self._cancelled:
            self.status.emit("swarm", "cancelled", "user ne rok diya")
            return SwarmResult(
                output="Cancelled by user.",
                worker_results=[],
                worker_types_used=worker_types,
                consensus_algorithm=self.consensus.algorithm,
            )

        # Step 3: Execute ALL workers in parallel — with cancellation support
        self.status.emit("swarm", "racing", "saare agents lage hue hain...")
        coros = [w.execute(enriched_task, cancel_event=self._cancel_event) for w in workers]
        results: list[WorkerResult] = await self._gather_with_cancel(coros)

        # Record costs from worker results
        task_cost = self._record_costs(results, task)

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

        # Step 5.5: Record learning — what worked, what didn't
        for r in results:
            self.learner.record_execution(
                task=task,
                worker_type=r.worker_type.value,
                success=r.success,
                confidence=r.confidence,
                duration_ms=r.elapsed_ms,
                error=r.error,
            )
        self.learner.save()

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

        # Add assistant response to conversation history (persisted)
        async with self._history_lock:
            self._conversation_history.append({
                "role": "assistant",
                "content": final_output[:1000],
                "ts": time.time(),
                "project": self.config.project_dir,
            })
            self._save_history()

        # Learning bridge — embed task+result for semantic recall
        if best_result:
            self.learning_bridge.learn_from_task(
                task=task,
                result=final_output[:500],
                worker_type=best_result.worker_type.value,
                confidence=best_result.confidence,
                success=best_result.success,
            )
            self.learning_bridge.save()

        # Update knowledge graph with task relationships
        task_node = f"task-{hash(task) % 100000}"
        self.knowledge_graph.add_entity(task_node, "task", description=task[:200])
        for wt in worker_types:
            worker_node = f"worker-{wt.value}"
            self.knowledge_graph.add_entity(worker_node, "worker_type")
            self.knowledge_graph.add_relationship(
                task_node, worker_node, "routed_to", weight=quality
            )
        if best_result:
            best_node = f"worker-{best_result.worker_type.value}"
            self.knowledge_graph.add_relationship(
                task_node, best_node, "best_solved_by", weight=best_result.confidence
            )
        self.knowledge_graph.save()

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

        # Aggregate token usage and cost
        total_input = sum(r.metadata.get("input_tokens", 0) for r in self.worker_results)
        total_output = sum(r.metadata.get("output_tokens", 0) for r in self.worker_results)
        total_cost = sum(
            CostTracker.calculate_cost(
                self._resolve_model(r),
                r.metadata.get("input_tokens", 0),
                r.metadata.get("output_tokens", 0),
            )
            for r in self.worker_results
        )

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
            "input_tokens": total_input,
            "output_tokens": total_output,
            "task_cost_usd": total_cost,
        }

    @staticmethod
    def _resolve_model(result: WorkerResult) -> str:
        """Best-effort model name from worker_id."""
        return result.metadata.get("model", "")
