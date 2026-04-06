"""Automatic task decomposition with JSON-structured LLM output.

Uses structured JSON responses instead of brittle line-by-line parsing. Validates
the resulting DAG (no cycles, no dangling deps, no empty subtasks) and asks the
LLM to re-plan if validation fails.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum

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
    dependencies: list[str] = field(default_factory=list)
    priority: int = 1  # 1=highest
    estimated_complexity: str = "medium"  # low, medium, high
    status: str = "pending"


@dataclass
class TaskPlan:
    """A complete decomposition plan."""

    original_task: str
    subtasks: list[SubTask]
    parallel_groups: list[list[str]]
    critical_path: list[str]

    @property
    def domains_involved(self) -> set[TaskDomain]:
        return {st.domain for st in self.subtasks}


class DecompositionError(Exception):
    """Raised when the LLM produces an invalid decomposition that can't be repaired."""


DECOMPOSITION_PROMPT = """You are a task decomposition expert. Your job is to break software engineering tasks into a structured plan.

You MUST respond with a single valid JSON object — no markdown, no prose, no code fences. The JSON schema is:

{
  "subtasks": [
    {
      "id": "unique-kebab-case-id",
      "title": "short title",
      "description": "detailed description of what needs to be done",
      "domain": "core|security|integration|support|research",
      "dependencies": ["id-of-prereq", ...],
      "priority": 1,
      "complexity": "low|medium|high"
    }
  ],
  "parallel_groups": [
    ["id1", "id2"],
    ["id3"]
  ],
  "critical_path": ["id1", "id3", "id4"]
}

Rules:
- Produce at least 3 subtasks for non-trivial tasks; more for large tasks.
- Every dependency id must refer to another subtask's id (no dangling refs).
- Dependencies must form a DAG (no cycles).
- parallel_groups should group subtasks that have no dependencies between them.
- critical_path is the longest dependency chain from entry to exit.
- priority: 1=must-have, 2=high, 3=normal, 4=nice-to-have, 5=optional.
- Start with research/exploration subtasks when the task is unclear.
- Always include support subtasks for tests and docs.

Return ONLY the JSON object. Do not wrap it in code fences."""


class TaskDecomposer:
    """Decomposes complex tasks into structured subtask plans."""

    def __init__(self, model: str = "anthropic:claude-sonnet-4-6"):
        self.model = model
        self._agent = create_deep_agent(
            model=model,
            system_prompt=DECOMPOSITION_PROMPT,
        )

    async def decompose(self, task: str, max_retries: int = 2) -> TaskPlan:
        """Decompose a task into a validated structured plan.

        Retries with corrective feedback if the LLM output fails validation.
        """
        last_error: str | None = None
        for attempt in range(max_retries + 1):
            prompt = f"Decompose this task:\n\n{task}"
            if last_error:
                prompt += (
                    f"\n\nYour previous attempt failed validation: {last_error}\n"
                    f"Fix the issue and respond with valid JSON only."
                )

            result = await self._agent.ainvoke(
                {"messages": [{"role": "user", "content": prompt}]}
            )
            output = result["messages"][-1].content
            if isinstance(output, list):
                # Some providers return a list of content blocks
                output = "".join(
                    (item.get("text", "") if isinstance(item, dict) else str(item))
                    for item in output
                )

            try:
                plan = self._parse_json_plan(task, output)
                self._validate_plan(plan)
                return plan
            except DecompositionError as e:
                last_error = str(e)
                logger.warning(f"Decomposition attempt {attempt + 1} failed: {e}")

        raise DecompositionError(
            f"Failed to produce a valid decomposition after {max_retries + 1} attempts. "
            f"Last error: {last_error}"
        )

    # ------------------------------------------------------------------
    # Parsing

    @staticmethod
    def _extract_json(text: str) -> dict:
        """Pull a JSON object out of LLM output, tolerant of code fences/prose."""
        # Strip markdown code fences
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fenced:
            text = fenced.group(1)

        # Find the first balanced JSON object
        start = text.find("{")
        if start < 0:
            raise DecompositionError("No JSON object found in LLM output")

        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    blob = text[start : i + 1]
                    try:
                        return json.loads(blob)
                    except json.JSONDecodeError as e:
                        raise DecompositionError(f"Malformed JSON: {e}") from e

        raise DecompositionError("Unclosed JSON object in LLM output")

    def _parse_json_plan(self, original_task: str, output: str) -> TaskPlan:
        data = self._extract_json(output)

        raw_subtasks = data.get("subtasks", [])
        if not isinstance(raw_subtasks, list) or not raw_subtasks:
            raise DecompositionError("'subtasks' must be a non-empty list")

        subtasks: list[SubTask] = []
        for i, raw in enumerate(raw_subtasks):
            if not isinstance(raw, dict):
                raise DecompositionError(f"subtasks[{i}] is not an object")
            try:
                subtasks.append(self._build_subtask(raw, i))
            except (KeyError, ValueError) as e:
                raise DecompositionError(f"subtasks[{i}] invalid: {e}") from e

        parallel_groups = data.get("parallel_groups", []) or []
        if not isinstance(parallel_groups, list):
            parallel_groups = []
        parallel_groups = [
            [str(i) for i in g if i] for g in parallel_groups if isinstance(g, list)
        ]

        critical_path = data.get("critical_path", []) or []
        if not isinstance(critical_path, list):
            critical_path = []
        critical_path = [str(i) for i in critical_path if i]

        return TaskPlan(
            original_task=original_task,
            subtasks=subtasks,
            parallel_groups=parallel_groups,
            critical_path=critical_path,
        )

    @staticmethod
    def _build_subtask(data: dict, fallback_idx: int) -> SubTask:
        subtask_id = str(data.get("id", "")).strip() or f"subtask-{fallback_idx + 1}"
        title = str(data.get("title", "")).strip()
        description = str(data.get("description", "")).strip()
        if not title or not description:
            raise ValueError("subtask must have non-empty title and description")

        domain_str = str(data.get("domain", "core")).strip().lower()
        try:
            domain = TaskDomain(domain_str)
        except ValueError:
            domain = TaskDomain.CORE

        raw_deps = data.get("dependencies", []) or []
        if isinstance(raw_deps, str):
            raw_deps = [d.strip() for d in raw_deps.split(",") if d.strip()]
        deps = [str(d).strip() for d in raw_deps if d]

        try:
            priority = int(data.get("priority", 3))
        except (TypeError, ValueError):
            priority = 3
        priority = max(1, min(5, priority))

        complexity = str(data.get("complexity", "medium")).strip().lower()
        if complexity not in ("low", "medium", "high"):
            complexity = "medium"

        return SubTask(
            id=subtask_id,
            title=title,
            description=description,
            domain=domain,
            dependencies=deps,
            priority=priority,
            estimated_complexity=complexity,
        )

    # ------------------------------------------------------------------
    # Validation

    @staticmethod
    def _validate_plan(plan: TaskPlan) -> None:
        """Ensure the plan has no dangling deps, no cycles, no duplicates."""
        ids = [st.id for st in plan.subtasks]
        id_set = set(ids)

        # Duplicates
        if len(ids) != len(id_set):
            dupes = [i for i in ids if ids.count(i) > 1]
            raise DecompositionError(f"Duplicate subtask ids: {sorted(set(dupes))}")

        # Dangling deps
        for st in plan.subtasks:
            for dep in st.dependencies:
                if dep not in id_set:
                    raise DecompositionError(
                        f"subtask '{st.id}' depends on unknown id '{dep}'"
                    )
                if dep == st.id:
                    raise DecompositionError(f"subtask '{st.id}' depends on itself")

        # Cycle detection via DFS
        color: dict[str, int] = {i: 0 for i in ids}  # 0=white, 1=gray, 2=black
        adj = {st.id: st.dependencies for st in plan.subtasks}

        def visit(node: str) -> None:
            if color[node] == 2:
                return
            if color[node] == 1:
                raise DecompositionError(f"Dependency cycle detected at '{node}'")
            color[node] = 1
            for nxt in adj.get(node, []):
                visit(nxt)
            color[node] = 2

        for node in ids:
            visit(node)
