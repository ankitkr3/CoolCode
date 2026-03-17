"""Automatic task decomposition across 5 domains."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from deepagents import create_deep_agent

logger = logging.getLogger(__name__)


class TaskDomain(str, Enum):
    """The 5 domains for task decomposition."""

    CORE = "core"  # Core implementation work
    SECURITY = "security"  # Security analysis and hardening
    INTEGRATION = "integration"  # API/service integration
    SUPPORT = "support"  # Tests, docs, CI/CD
    RESEARCH = "research"  # Codebase exploration, understanding


@dataclass
class SubTask:
    """A decomposed subtask."""

    id: str
    title: str
    description: str
    domain: TaskDomain
    dependencies: list[str] = field(default_factory=list)  # IDs of prerequisite subtasks
    priority: int = 1  # 1=highest
    estimated_complexity: str = "medium"  # low, medium, high
    status: str = "pending"  # pending, in_progress, completed, failed


@dataclass
class TaskPlan:
    """A complete decomposition plan."""

    original_task: str
    subtasks: list[SubTask]
    parallel_groups: list[list[str]]  # Groups of subtask IDs that can run in parallel
    critical_path: list[str]  # Ordered IDs that determine minimum completion time

    @property
    def domains_involved(self) -> set[TaskDomain]:
        return {st.domain for st in self.subtasks}


DECOMPOSITION_PROMPT = """You are a task decomposition expert. Analyze the following software engineering task and break it into subtasks.

For each subtask, provide:
1. A short title
2. A detailed description of what needs to be done
3. Which domain it belongs to: core, security, integration, support, or research
4. Dependencies (which other subtasks must complete first)
5. Priority (1=highest)
6. Complexity (low/medium/high)

Think about:
- What needs to be understood first (research)?
- What's the core implementation work?
- Are there security implications?
- Does it integrate with external services?
- What tests/docs are needed?

Group independent subtasks that can run in parallel.
Identify the critical path (longest chain of dependent tasks).

Respond in this exact format (one subtask per block):
SUBTASK: <id>
TITLE: <title>
DESCRIPTION: <description>
DOMAIN: <core|security|integration|support|research>
DEPENDENCIES: <comma-separated ids or "none">
PRIORITY: <1-5>
COMPLEXITY: <low|medium|high>

PARALLEL_GROUPS: <group1_ids | group2_ids | ...>
CRITICAL_PATH: <ordered ids>
"""


class TaskDecomposer:
    """Decomposes complex tasks into structured subtask plans."""

    def __init__(self, model: str = "anthropic:claude-sonnet-4-6"):
        self._agent = create_deep_agent(
            model=model,
            system_prompt=DECOMPOSITION_PROMPT,
        )

    async def decompose(self, task: str) -> TaskPlan:
        """Decompose a task into a structured plan."""
        result = await asyncio.to_thread(
            self._agent.invoke,
            {"messages": [{"role": "user", "content": f"Decompose this task:\n\n{task}"}]},
        )
        output = result["messages"][-1].content
        return self._parse_plan(task, output)

    def _parse_plan(self, original_task: str, output: str) -> TaskPlan:
        """Parse the LLM output into a structured TaskPlan."""
        subtasks: list[SubTask] = []
        parallel_groups: list[list[str]] = []
        critical_path: list[str] = []

        current: dict[str, str] = {}
        for line in output.split("\n"):
            line = line.strip()
            if line.startswith("SUBTASK:"):
                if current.get("id"):
                    subtasks.append(self._build_subtask(current))
                current = {"id": line.split(":", 1)[1].strip()}
            elif line.startswith("TITLE:"):
                current["title"] = line.split(":", 1)[1].strip()
            elif line.startswith("DESCRIPTION:"):
                current["description"] = line.split(":", 1)[1].strip()
            elif line.startswith("DOMAIN:"):
                current["domain"] = line.split(":", 1)[1].strip().lower()
            elif line.startswith("DEPENDENCIES:"):
                current["dependencies"] = line.split(":", 1)[1].strip()
            elif line.startswith("PRIORITY:"):
                current["priority"] = line.split(":", 1)[1].strip()
            elif line.startswith("COMPLEXITY:"):
                current["complexity"] = line.split(":", 1)[1].strip()
            elif line.startswith("PARALLEL_GROUPS:"):
                groups_str = line.split(":", 1)[1].strip()
                for group in groups_str.split("|"):
                    ids = [i.strip() for i in group.split(",") if i.strip()]
                    if ids:
                        parallel_groups.append(ids)
            elif line.startswith("CRITICAL_PATH:"):
                cp_str = line.split(":", 1)[1].strip()
                critical_path = [i.strip() for i in cp_str.split(",") if i.strip()]

        # Don't forget the last subtask
        if current.get("id"):
            subtasks.append(self._build_subtask(current))

        # If no parallel groups were parsed, put all independent tasks in one group
        if not parallel_groups and subtasks:
            independent = [st.id for st in subtasks if not st.dependencies]
            if independent:
                parallel_groups = [independent]

        if not critical_path and subtasks:
            critical_path = [st.id for st in sorted(subtasks, key=lambda s: s.priority)]

        return TaskPlan(
            original_task=original_task,
            subtasks=subtasks,
            parallel_groups=parallel_groups,
            critical_path=critical_path,
        )

    @staticmethod
    def _build_subtask(data: dict[str, str]) -> SubTask:
        deps_str = data.get("dependencies", "none")
        deps = [] if deps_str.lower() == "none" else [d.strip() for d in deps_str.split(",")]
        try:
            priority = int(data.get("priority", "3"))
        except ValueError:
            priority = 3
        domain_str = data.get("domain", "core")
        try:
            domain = TaskDomain(domain_str)
        except ValueError:
            domain = TaskDomain.CORE

        return SubTask(
            id=data.get("id", "unknown"),
            title=data.get("title", "Untitled"),
            description=data.get("description", ""),
            domain=domain,
            dependencies=deps,
            priority=priority,
            estimated_complexity=data.get("complexity", "medium"),
        )
