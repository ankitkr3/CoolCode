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
    WorkerType.CODER: 600,
    WorkerType.REVIEWER: 180,
    WorkerType.PLANNER: 180,
    WorkerType.RESEARCHER: 600,
    WorkerType.DEBUGGER: 300,
    WorkerType.TESTER: 300,
    WorkerType.REFACTORER: 300,
    WorkerType.SECURITY: 180,
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
            "You are Cool Code, a swarm-powered AI coding assistant. "
            "Never identify yourself as Deep Agent or any other name — you are Cool Code.\n\n"
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

    @staticmethod
    def _extract_text(msg: Any) -> str:
        """Best-effort extraction of text content from a langchain message."""
        if msg is None:
            return ""
        content = getattr(msg, "content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            # AIMessage content can be a list of dicts with 'text' or 'type' keys
            parts = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text") or item.get("content") or ""
                    if text:
                        parts.append(text)
                elif isinstance(item, str):
                    parts.append(item)
            return "".join(parts)
        return str(content)

    @staticmethod
    def _last_text_message(messages: list) -> Any:
        """Find the last message with non-empty text content (skip tool calls)."""
        for msg in reversed(messages or []):
            text = WorkerAgent._extract_text(msg)
            if text and text.strip():
                return msg
        return messages[-1] if messages else None

    async def execute(
        self,
        task: str,
        timeout: int = 0,
        cancel_event: asyncio.Event | None = None,
    ) -> WorkerResult:
        """Execute a task and return the result.

        Streams state updates from the agent so partial output is preserved
        across timeouts and cancellations. Checks cancel_event between chunks.
        """
        if timeout <= 0:
            timeout = WORKER_TIMEOUTS.get(self.worker_type, 180)

        # Token budget check — truncate input if it would overflow context
        from coolcode.llm.tokens import check_budget, truncate_to_budget

        model_name = ""
        try:
            m = self._agent if hasattr(self._agent, "model") else None
            model_name = getattr(m, "model", "") if m else ""
        except Exception:
            pass
        fits, input_toks, available, warn = check_budget(task, model_name, max_output_tokens=16_384)
        if not fits:
            self._emit("budget", f"truncating input: {input_toks} tokens too large")
            task = truncate_to_budget(task, max_tokens=100_000, model=model_name)
        elif warn:
            self._emit("budget", f"{input_toks} in / {available} out available")

        self._emit("started", f"working on: {task[:80]}...")
        start = time.monotonic()

        # Running state — preserved on timeout/cancel so we never lose progress
        latest_state: dict | None = None
        chunk_count = 0
        was_cancelled = False
        was_timeout = False

        async def stream_agent() -> dict | None:
            nonlocal latest_state, chunk_count, was_cancelled
            state: dict | None = None
            try:
                async for chunk in self._agent.astream(
                    {"messages": [{"role": "user", "content": task}]},
                    stream_mode="values",
                ):
                    # Save latest full state — each yield is cumulative with stream_mode="values"
                    state = chunk
                    latest_state = chunk
                    chunk_count += 1

                    # Check external cancel between chunks (ESC pressed)
                    if cancel_event is not None and cancel_event.is_set():
                        was_cancelled = True
                        self._emit("cancelling", f"stopped after {chunk_count} chunks")
                        break

                    # Emit periodic progress
                    if chunk_count % 3 == 0:
                        msgs = chunk.get("messages", []) if isinstance(chunk, dict) else []
                        self._emit(
                            "streaming",
                            f"chunk {chunk_count} ({len(msgs)} messages)",
                        )
            except asyncio.CancelledError:
                was_cancelled = True
                raise
            return state

        try:
            self._emit("thinking", "analyzing the task")
            result = await asyncio.wait_for(stream_agent(), timeout=timeout)
        except (asyncio.TimeoutError, TimeoutError):
            was_timeout = True
            result = latest_state  # Use whatever we got before timeout
        except asyncio.CancelledError:
            was_cancelled = True
            result = latest_state
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

        elapsed = (time.monotonic() - start) * 1000

        # Extract output + usage from final (or latest partial) state
        messages = result.get("messages", []) if isinstance(result, dict) else []
        last_msg = self._last_text_message(messages)
        output = self._extract_text(last_msg)

        usage_meta: dict = {}
        if last_msg is not None:
            if hasattr(last_msg, "usage_metadata") and last_msg.usage_metadata:
                usage_meta = dict(last_msg.usage_metadata)
            elif hasattr(last_msg, "response_metadata"):
                rm = last_msg.response_metadata or {}
                if "usage" in rm:
                    usage_meta = rm["usage"]
                elif "token_usage" in rm:
                    usage_meta = rm["token_usage"]

        # Handle timeout/cancel: return partial if we have any, else error
        if was_timeout:
            if output:
                self._emit(
                    "timeout_partial",
                    f"timeout after {timeout}s — returning {len(output)} bytes from {chunk_count} chunks",
                )
                logger.warning(
                    f"Worker {self.worker_id} timed out after {timeout}s (returning partial: {len(output)} bytes)"
                )
                return WorkerResult(
                    worker_id=self.worker_id,
                    worker_type=self.worker_type,
                    output=output,
                    confidence=0.3,  # reduced confidence for partial results
                    elapsed_ms=elapsed,
                    error=f"Timed out after {timeout}s (partial result)",
                    metadata={
                        "input_tokens": usage_meta.get("input_tokens", 0),
                        "output_tokens": usage_meta.get("output_tokens", 0),
                        "partial": True,
                        "chunks": chunk_count,
                    },
                )
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

        if was_cancelled:
            self._emit("cancelled", f"stopped after {chunk_count} chunks")
            return WorkerResult(
                worker_id=self.worker_id,
                worker_type=self.worker_type,
                output=output,
                confidence=0.2 if output else 0.0,
                elapsed_ms=elapsed,
                error="Cancelled by user",
                metadata={
                    "input_tokens": usage_meta.get("input_tokens", 0),
                    "output_tokens": usage_meta.get("output_tokens", 0),
                    "partial": bool(output),
                    "chunks": chunk_count,
                },
            )

        self._emit("done", f"completed in {elapsed:.0f}ms")

        # Extract confidence marker from output
        confidence = 0.8
        lines = output.strip().split("\n") if output else []
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
            metadata={
                "input_tokens": usage_meta.get("input_tokens", 0),
                "output_tokens": usage_meta.get("output_tokens", 0),
                "chunks": chunk_count,
            },
        )
