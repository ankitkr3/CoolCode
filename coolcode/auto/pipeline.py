"""Autonomous pipeline orchestration.

Decomposes a task into stages, executes them with optional human gates,
and manages git checkpoints for rollback.

Usage:
    /auto "build user dashboard with auth and tests"
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from coolcode.agent.swarm import Swarm
from coolcode.agent.worker import WorkerType
from coolcode.auto.checkpoints import CheckpointManager
from coolcode.auto.gates import GateDecision, GateManager, GateResult
from coolcode.cost import CostTracker

logger = logging.getLogger(__name__)

console = Console()


@dataclass
class PipelineStage:
    """A single stage in an autonomous pipeline."""

    description: str
    worker_types: list[WorkerType]
    prompt: str
    is_gate: bool = False  # pause for human approval after this stage
    estimated_cost: float = 0.0
    timeout: int = 600


@dataclass
class PipelinePlan:
    """A planned autonomous pipeline."""

    task: str
    stages: list[PipelineStage]
    estimated_total_cost: float = 0.0
    gate_frequency: int = 2  # gate every N stages (0 = no gates, 1 = every stage)

    @property
    def total_stages(self) -> int:
        return len(self.stages)


@dataclass
class StageResult:
    """Result of a single pipeline stage."""

    stage_index: int
    description: str
    output: str
    cost_usd: float
    elapsed_ms: float
    success: bool


@dataclass
class AutoResult:
    """Result of an autonomous pipeline execution."""

    task: str
    stages_completed: int
    total_stages: int
    stage_results: list[StageResult]
    total_cost: float
    final_output: str
    cancelled: bool = False
    rollback_stage: int = -1


# Default stage templates for auto-decomposition
STAGE_TEMPLATES = [
    PipelineStage(
        description="Plan & Architecture",
        worker_types=[WorkerType.PLANNER, WorkerType.RESEARCHER],
        prompt=(
            "Analyze the task and create a detailed implementation plan.\n"
            "1. Understand what needs to be built\n"
            "2. Research the existing codebase for relevant patterns\n"
            "3. Identify files to create/modify\n"
            "4. Create a step-by-step plan with clear acceptance criteria"
        ),
        is_gate=True,  # always gate after planning
    ),
    PipelineStage(
        description="Implement Core Logic",
        worker_types=[WorkerType.CODER],
        prompt=(
            "Implement the core logic based on the plan from the previous stage.\n"
            "Follow existing patterns. Write clean, minimal code."
        ),
        timeout=300,
    ),
    PipelineStage(
        description="Write Tests",
        worker_types=[WorkerType.TESTER],
        prompt=(
            "Write comprehensive tests for the implementation from the previous stage.\n"
            "Cover happy paths, edge cases, and error conditions."
        ),
    ),
    PipelineStage(
        description="Code Review",
        worker_types=[WorkerType.REVIEWER, WorkerType.SECURITY],
        prompt=(
            "Review the implementation and tests from previous stages.\n"
            "Check for: correctness, security, performance, style.\n"
            "Suggest specific improvements."
        ),
        is_gate=True,  # gate after review
    ),
    PipelineStage(
        description="Apply Fixes",
        worker_types=[WorkerType.CODER],
        prompt=(
            "Apply the fixes and improvements suggested in the code review.\n"
            "Only fix issues that were flagged — don't make other changes."
        ),
    ),
]


class AutoPipeline:
    """Orchestrates autonomous multi-stage pipelines.

    Flow:
    1. Plan: decompose task into stages (or use templates)
    2. Show plan to user with cost estimates
    3. Execute stages sequentially with:
       - Git checkpoints before each stage
       - Human gates at configurable intervals
       - Cost tracking throughout
    4. Return final result
    """

    def __init__(
        self,
        swarm: Swarm,
        gate_frequency: int = 2,  # gate every N stages (0=none, 1=every)
    ):
        self.swarm = swarm
        self.gate_frequency = gate_frequency
        self.gate_manager = GateManager()
        self.checkpoint_manager = CheckpointManager(swarm.config.project_dir)

    def plan(self, task: str) -> PipelinePlan:
        """Create a pipeline plan for the given task.

        Uses default stage templates and customizes prompts with the task.
        """
        stages = []
        for template in STAGE_TEMPLATES:
            stage = PipelineStage(
                description=template.description,
                worker_types=list(template.worker_types),
                prompt=f"[TASK]: {task}\n\n{template.prompt}",
                is_gate=template.is_gate,
                estimated_cost=self._estimate_stage_cost(template),
                timeout=template.timeout,
            )
            # Apply gate frequency
            if self.gate_frequency > 0:
                stage_idx = len(stages)
                if (stage_idx + 1) % self.gate_frequency == 0:
                    stage.is_gate = True
            stages.append(stage)

        total_cost = sum(s.estimated_cost for s in stages)

        return PipelinePlan(
            task=task,
            stages=stages,
            estimated_total_cost=total_cost,
            gate_frequency=self.gate_frequency,
        )

    def show_plan(self, plan: PipelinePlan) -> None:
        """Display the pipeline plan for user confirmation."""
        table = Table(
            title="[bold]Autonomous Pipeline Plan[/bold]",
            show_header=True,
            border_style="cyan",
        )
        table.add_column("#", style="dim", width=4)
        table.add_column("Stage", style="bold")
        table.add_column("Workers", style="cyan")
        table.add_column("Gate", style="yellow")
        table.add_column("Est. Cost", style="green")

        for i, stage in enumerate(plan.stages):
            workers = ", ".join(wt.value for wt in stage.worker_types)
            gate = "GATE" if stage.is_gate else ""
            table.add_row(
                str(i + 1),
                stage.description,
                workers,
                gate,
                f"${stage.estimated_cost:.3f}",
            )

        console.print(table)
        console.print(f"\n[bold]Total stages:[/bold] {plan.total_stages}")
        console.print(f"[bold]Estimated cost:[/bold] ${plan.estimated_total_cost:.3f}")
        console.print(f"[bold]Gates:[/bold] {sum(1 for s in plan.stages if s.is_gate)} approval points")
        console.print()

    async def execute(self, plan: PipelinePlan) -> AutoResult:
        """Execute the pipeline with gates and checkpoints."""
        stage_results: list[StageResult] = []
        all_stage_outputs: list[str] = []
        total_cost = 0.0

        for stage_idx, stage in enumerate(plan.stages):
            # Check cancellation
            if self.swarm._cancelled:
                break

            # Create checkpoint before stage
            self.checkpoint_manager.create(
                name=stage.description.lower().replace(" ", "-"),
                stage_index=stage_idx,
            )

            self.swarm.status.emit(
                "auto", f"stage {stage_idx + 1}/{plan.total_stages}",
                stage.description,
            )

            # Build context with previous stage outputs
            context = stage.prompt
            for i, output in enumerate(all_stage_outputs):
                context += f"\n\n--- STAGE {i + 1} RESULTS ---\n{output[:4000]}\n--- END ---"

            start = time.monotonic()

            # Set the swarm goal temporarily and execute
            old_goal = self.swarm.goal
            self.swarm.set_goal("general")

            try:
                result = await self.swarm.execute(context)
                output = result.output
                success = bool(result.best_worker and result.best_worker.confidence > 0.3)
                stage_cost = result.stats.get("task_cost_usd", 0.0)
            except Exception as e:
                output = f"Error: {e}"
                success = False
                stage_cost = 0.0

            self.swarm.set_goal(old_goal)
            elapsed = (time.monotonic() - start) * 1000
            total_cost += stage_cost

            stage_result = StageResult(
                stage_index=stage_idx,
                description=stage.description,
                output=output,
                cost_usd=stage_cost,
                elapsed_ms=elapsed,
                success=success,
            )
            stage_results.append(stage_result)
            all_stage_outputs.append(output)

            self.swarm.status.emit(
                "auto", "stage done",
                f"{stage.description} — ${stage_cost:.4f}, {elapsed:.0f}ms"
            )

            # Human gate
            if stage.is_gate and not self.swarm._cancelled:
                gate_result = self.gate_manager.prompt_gate(
                    stage_index=stage_idx,
                    stage_label=stage.description,
                    stage_output=output,
                    total_stages=plan.total_stages,
                    cost_so_far=total_cost,
                )

                if gate_result.decision == GateDecision.REJECT:
                    return AutoResult(
                        task=plan.task,
                        stages_completed=stage_idx + 1,
                        total_stages=plan.total_stages,
                        stage_results=stage_results,
                        total_cost=total_cost,
                        final_output="\n\n".join(all_stage_outputs),
                        cancelled=True,
                    )

                elif gate_result.decision == GateDecision.ROLLBACK:
                    self.checkpoint_manager.rollback_to_stage(stage_idx)
                    # Remove this stage's results
                    stage_results.pop()
                    all_stage_outputs.pop()
                    return AutoResult(
                        task=plan.task,
                        stages_completed=stage_idx,
                        total_stages=plan.total_stages,
                        stage_results=stage_results,
                        total_cost=total_cost,
                        final_output="\n\n".join(all_stage_outputs),
                        rollback_stage=stage_idx,
                    )

                elif gate_result.decision == GateDecision.EDIT:
                    # Re-run this stage with corrective instructions
                    stage.prompt += f"\n\nCORRECTIVE INSTRUCTIONS: {gate_result.edit_instructions}"
                    # The loop will continue but we don't go back — the corrective
                    # context will be picked up by the next stage

        # Cleanup checkpoints on success
        if not any(sr.success is False for sr in stage_results):
            self.checkpoint_manager.cleanup()

        return AutoResult(
            task=plan.task,
            stages_completed=len(stage_results),
            total_stages=plan.total_stages,
            stage_results=stage_results,
            total_cost=total_cost,
            final_output="\n\n".join(all_stage_outputs),
        )

    def _estimate_stage_cost(self, stage: PipelineStage) -> float:
        """Rough cost estimate for a stage based on model and workers."""
        # Rough estimate: ~2K input tokens, ~1K output tokens per worker
        model = ""
        if self.swarm.config.providers:
            model = self.swarm.config.providers[0].model
        per_worker = CostTracker.calculate_cost(model, 2000, 1000)
        return per_worker * len(stage.worker_types)
