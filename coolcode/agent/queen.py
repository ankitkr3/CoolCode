"""Queen agent — the coordinator of the swarm.

3 queen types:
- Coordinator: orchestrates workers, merges results
- Evaluator: judges worker outputs, picks the best
- Delegator: routes tasks to the right worker types
"""

from __future__ import annotations

import asyncio
import logging
from enum import Enum
from typing import Any

from deepagents import create_deep_agent
from langchain_core.language_models import BaseChatModel

from coolcode.agent.consensus import ConsensusEngine, ConsensusResult, Vote
from coolcode.agent.worker import WorkerAgent, WorkerResult, WorkerType

logger = logging.getLogger(__name__)


class QueenType(str, Enum):
    COORDINATOR = "coordinator"
    EVALUATOR = "evaluator"
    DELEGATOR = "delegator"


QUEEN_PROMPTS: dict[QueenType, str] = {
    QueenType.COORDINATOR: (
        "You are the Queen Coordinator of a coding agent swarm. Your role is to:\n"
        "1. Analyze incoming tasks and determine which worker types are needed\n"
        "2. Decompose complex tasks into subtasks for parallel execution\n"
        "3. Merge and synthesize results from multiple workers into a coherent output\n"
        "4. Resolve conflicts between worker outputs using consensus\n"
        "5. Ensure the final output is complete, correct, and high quality\n\n"
        "Think strategically. Prefer parallel execution when subtasks are independent. "
        "Always validate that merged outputs are consistent."
    ),
    QueenType.EVALUATOR: (
        "You are the Queen Evaluator. Multiple worker agents have produced candidate solutions. "
        "Your role is to:\n"
        "1. Evaluate each candidate on correctness, completeness, code quality, and efficiency\n"
        "2. Identify the strengths and weaknesses of each approach\n"
        "3. Either select the best candidate or synthesize a superior solution from the best parts\n"
        "4. Provide clear reasoning for your evaluation\n\n"
        "Be rigorous. The best solution isn't always the most complex one."
    ),
    QueenType.DELEGATOR: (
        "You are the Queen Delegator. Your role is intelligent task routing:\n"
        "1. Analyze the incoming task's nature (coding, debugging, testing, security, etc.)\n"
        "2. Determine which worker type(s) are best suited\n"
        "3. If the task spans multiple domains, decompose and route to multiple workers\n"
        "4. Estimate complexity and decide if parallel racing or sequential pipeline is better\n\n"
        "Output your routing decision as a structured plan."
    ),
}


class QueenAgent:
    """The queen coordinates worker agents, evaluates outputs, and reaches consensus."""

    def __init__(
        self,
        queen_type: QueenType = QueenType.COORDINATOR,
        model: BaseChatModel | str = "anthropic:claude-sonnet-4-6",
        consensus: ConsensusEngine | None = None,
    ):
        self.queen_type = queen_type
        self.consensus = consensus or ConsensusEngine(algorithm="weighted")

        agent_kwargs: dict[str, Any] = {
            "system_prompt": QUEEN_PROMPTS[queen_type],
        }
        if isinstance(model, str):
            agent_kwargs["model"] = model
        else:
            agent_kwargs["model"] = model

        self._agent = create_deep_agent(**agent_kwargs)

    def decide_worker_types(self, task: str) -> list[WorkerType]:
        """Analyze a task and determine which worker types should handle it."""
        task_lower = task.lower()

        types: list[WorkerType] = []

        # Keyword-based routing (fast path before LLM call)
        routing_rules = {
            WorkerType.DEBUGGER: ["bug", "fix", "error", "crash", "broken", "failing", "debug"],
            WorkerType.TESTER: ["test", "spec", "coverage", "assert"],
            WorkerType.SECURITY: ["security", "vulnerability", "cve", "owasp", "auth", "injection"],
            WorkerType.REFACTORER: ["refactor", "clean up", "simplify", "restructure"],
            WorkerType.REVIEWER: ["review", "audit", "check", "inspect"],
            WorkerType.RESEARCHER: ["search", "find", "understand", "explain", "how does"],
            WorkerType.PLANNER: ["plan", "design", "architect", "break down"],
        }

        for worker_type, keywords in routing_rules.items():
            if any(kw in task_lower for kw in keywords):
                types.append(worker_type)

        # Default: always include a coder for implementation tasks
        if not types or any(
            kw in task_lower
            for kw in ["implement", "create", "add", "build", "write", "make", "change", "update"]
        ):
            types.append(WorkerType.CODER)

        return list(dict.fromkeys(types))  # deduplicate preserving order

    async def evaluate_results(self, results: list[WorkerResult]) -> ConsensusResult:
        """Evaluate multiple worker results using consensus."""
        successful = [r for r in results if r.success]
        if not successful:
            return ConsensusResult(
                algorithm=self.consensus.algorithm,
                winner=None,
                votes=[],
                agreement_ratio=0.0,
                converged=False,
                details={"error": "All workers failed"},
            )

        if len(successful) == 1:
            r = successful[0]
            vote = Vote(agent_id=r.worker_id, choice=r.worker_id, confidence=r.confidence)
            return ConsensusResult(
                algorithm="single",
                winner=r.worker_id,
                votes=[vote],
                agreement_ratio=1.0,
                converged=True,
            )

        # For parallel racing: each result is a candidate, queen evaluates
        votes: list[Vote] = []
        for r in successful:
            votes.append(
                Vote(
                    agent_id=r.worker_id,
                    choice=r.worker_id,
                    confidence=r.confidence,
                    reasoning=r.output[:200],
                )
            )

        return self.consensus.resolve(votes)

    async def merge_results(self, task: str, results: list[WorkerResult]) -> str:
        """Use the queen's LLM to merge multiple worker outputs into one."""
        successful = [r for r in results if r.success]
        if not successful:
            return "All workers failed to produce a result."
        if len(successful) == 1:
            return successful[0].output

        merge_prompt = (
            f"Original task: {task}\n\n"
            "The following workers produced these results. "
            "Merge them into a single, optimal output:\n\n"
        )
        for r in successful:
            merge_prompt += (
                f"--- Worker: {r.worker_id} (type: {r.worker_type.value}, "
                f"confidence: {r.confidence:.2f}) ---\n{r.output}\n\n"
            )
        merge_prompt += (
            "Produce the best possible merged output. If the outputs conflict, "
            "prefer the higher-confidence one unless you can see it's wrong."
        )

        result = await asyncio.to_thread(
            self._agent.invoke,
            {"messages": [{"role": "user", "content": merge_prompt}]},
        )
        return result["messages"][-1].content
