"""Worker agents — 8 specialized types for the swarm."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from deepagents import create_deep_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)


class WorkerType(str, Enum):
    """The 8 worker specializations."""

    CODER = "coder"  # Writes and edits code
    REVIEWER = "reviewer"  # Reviews code for quality, bugs, security
    PLANNER = "planner"  # Breaks tasks into subtasks
    RESEARCHER = "researcher"  # Searches codebase and docs
    DEBUGGER = "debugger"  # Diagnoses and fixes bugs
    TESTER = "tester"  # Writes and runs tests
    REFACTORER = "refactorer"  # Improves code structure
    SECURITY = "security"  # Security analysis and hardening


# System prompt fragments per worker type
WORKER_PROMPTS: dict[WorkerType, str] = {
    WorkerType.CODER: (
        "You are an expert software engineer. Write clean, efficient, production-ready code. "
        "Follow existing patterns in the codebase. Keep changes minimal and focused. "
        "Think step-by-step: understand the requirement, examine existing code, plan your change, implement it.\n"
        "Be fast: read only the files you need to modify. Use grep to find relevant code, don't read everything."
    ),
    WorkerType.REVIEWER: (
        "You are a senior code reviewer. Analyze code for correctness, performance, security, "
        "readability, and adherence to best practices. Be specific and actionable in your feedback. "
        "Prioritize issues by severity: critical bugs > security > performance > style."
    ),
    WorkerType.PLANNER: (
        "You are a software architect specializing in task decomposition. Break complex tasks into "
        "clear, ordered subtasks. Identify dependencies, risks, and the critical path. "
        "Each subtask should be independently completable and testable."
    ),
    WorkerType.RESEARCHER: (
        "You are a codebase researcher. Your job is to quickly understand code architecture.\n\n"
        "RULES FOR SPEED:\n"
        "1. Start with list_dir to see the top-level structure. NEVER use 'find' shell commands.\n"
        "2. Read README.md or equivalent first if it exists.\n"
        "3. Only read KEY files: entry points (main.py, app.py, index.ts), config files, and core modules.\n"
        "4. Do NOT read every file. Read at most 5-8 files. Skim structure, don't read line by line.\n"
        "5. Use glob_search to find patterns (e.g., '**/*.py') instead of shell find commands.\n"
        "6. Use grep_search to understand imports/dependencies instead of reading whole files.\n"
        "7. Summarize as you go. Don't collect everything then summarize — stream your understanding.\n\n"
        "Be FAST. A good 80% understanding in 30 seconds beats a perfect understanding in 5 minutes."
    ),
    WorkerType.DEBUGGER: (
        "You are an expert debugger. Systematically diagnose issues using the scientific method: "
        "observe symptoms, form hypotheses, test them, narrow down root causes. "
        "Always verify your fix doesn't introduce new issues."
    ),
    WorkerType.TESTER: (
        "You are a testing specialist. Write comprehensive tests covering happy paths, edge cases, "
        "error conditions, and boundary values. Prefer integration tests over mocks. "
        "Ensure tests are deterministic and fast."
    ),
    WorkerType.REFACTORER: (
        "You are a refactoring specialist. Improve code structure without changing behavior. "
        "Focus on reducing duplication, improving naming, simplifying complex logic, and "
        "extracting clear abstractions. Always ensure tests still pass after refactoring."
    ),
    WorkerType.SECURITY: (
        "You are a security engineer. Analyze code for OWASP Top 10 vulnerabilities, injection "
        "flaws, authentication issues, data exposure, and misconfigurations. "
        "Provide specific, actionable remediation steps with code examples."
    ),
}


# Per-worker-type timeout in seconds
WORKER_TIMEOUTS: dict[WorkerType, int] = {
    WorkerType.CODER: 180,
    WorkerType.REVIEWER: 120,
    WorkerType.PLANNER: 90,
    WorkerType.RESEARCHER: 180,
    WorkerType.DEBUGGER: 180,
    WorkerType.TESTER: 180,
    WorkerType.REFACTORER: 180,
    WorkerType.SECURITY: 120,
}


@dataclass
class WorkerResult:
    """Result from a worker agent's execution."""

    worker_id: str
    worker_type: WorkerType
    output: str
    confidence: float = 0.8
    elapsed_ms: float = 0.0
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.error is None


class WorkerAgent:
    """A specialized worker agent in the swarm.

    Each worker is a Deep Agent configured with a type-specific system prompt
    and tools appropriate for its specialization.
    """

    def __init__(
        self,
        worker_id: str,
        worker_type: WorkerType,
        model: BaseChatModel | str = "anthropic:claude-sonnet-4-6",
        tools: list | None = None,
        extra_instructions: str = "",
        status_tracker: Any = None,
        system_prompt_override: str | None = None,
    ):
        self.worker_id = worker_id
        self.worker_type = worker_type
        self._status = status_tracker

        # Goal-specific prompts replace the default; otherwise use the standard prompt
        base_prompt = system_prompt_override or WORKER_PROMPTS[worker_type]

        system_prompt = (
            f"[Worker ID: {worker_id} | Type: {worker_type.value}]\n\n"
            f"{base_prompt}\n\n"
            f"{extra_instructions}\n\n"
            "Be fast and focused. Use tools efficiently — don't read files you don't need.\n"
            "Use write_todos to track multi-step work.\n"
            "After completing your task, rate your confidence in your output from 0.0 to 1.0 "
            "on the last line as: CONFIDENCE: <number>"
        )

        agent_kwargs: dict[str, Any] = {
            "system_prompt": system_prompt,
            "model": model,
        }
        if tools:
            agent_kwargs["tools"] = tools

        # Use Deep Agents' built-in middleware for planning, filesystem, and summarization
        # This gives us write_todos, read_file/write_file/edit_file, and context management for free
        self._agent = create_deep_agent(**agent_kwargs)

    def _emit(self, action: str, detail: str = "") -> None:
        if self._status:
            self._status.emit(self.worker_id, action, detail)

    async def execute(self, task: str, timeout: int = 0) -> WorkerResult:
        """Execute a task and return the result. Uses per-worker-type timeout if not specified."""
        if timeout <= 0:
            timeout = WORKER_TIMEOUTS.get(self.worker_type, 180)
        self._emit("started", f"working on: {task[:80]}...")
        start = time.monotonic()
        try:
            self._emit("thinking", "analyzing the task")
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    self._agent.invoke,
                    {"messages": [{"role": "user", "content": task}]},
                ),
                timeout=timeout,
            )
            output = result["messages"][-1].content
            elapsed = (time.monotonic() - start) * 1000

            self._emit("done", f"completed in {elapsed:.0f}ms")

            # Extract confidence from output
            confidence = 0.8
            lines = output.strip().split("\n")
            for line in reversed(lines):
                if line.strip().startswith("CONFIDENCE:"):
                    try:
                        confidence = float(line.split(":", 1)[1].strip())
                        confidence = max(0.0, min(1.0, confidence))
                    except ValueError:
                        pass
                    break

            return WorkerResult(
                worker_id=self.worker_id,
                worker_type=self.worker_type,
                output=output,
                confidence=confidence,
                elapsed_ms=elapsed,
            )
        except (asyncio.TimeoutError, TimeoutError):
            elapsed = (time.monotonic() - start) * 1000
            self._emit("timeout", f"timed out after {timeout}s")
            logger.warning(f"Worker {self.worker_id} timed out after {timeout}s")
            return WorkerResult(
                worker_id=self.worker_id,
                worker_type=self.worker_type,
                output="",
                confidence=0.0,
                elapsed_ms=elapsed,
                error=f"Timed out after {timeout}s",
            )
        except Exception as e:
            elapsed = (time.monotonic() - start) * 1000
            self._emit("failed", str(e)[:100])
            logger.error(f"Worker {self.worker_id} failed: {e}")
            return WorkerResult(
                worker_id=self.worker_id,
                worker_type=self.worker_type,
                output="",
                confidence=0.0,
                elapsed_ms=elapsed,
                error=str(e),
            )
