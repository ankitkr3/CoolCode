"""Self-learning system — learns from your workflow, adapts over time.

Tracks what works:
- Which worker types succeed for which task patterns
- Which prompts produce better confidence scores
- Which files are frequently accessed together
- User corrections and preferences

Uses these learnings to:
- Route future tasks more accurately
- Pre-fetch likely-needed files
- Adjust worker count based on task complexity
- Skip unnecessary workers
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TaskPattern:
    """A learned pattern from past task executions."""

    keywords: list[str]
    best_worker_type: str
    avg_confidence: float
    avg_duration_ms: float
    success_count: int = 0
    fail_count: int = 0
    last_used: float = field(default_factory=time.time)

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.fail_count
        return self.success_count / total if total > 0 else 0.0

    @property
    def score(self) -> float:
        """Composite score: success_rate * confidence * recency."""
        recency = 1.0 / (1.0 + (time.time() - self.last_used) / 86400)  # decay over days
        return self.success_rate * self.avg_confidence * recency


@dataclass
class FileAccessPattern:
    """Tracks which files are accessed together — for pre-fetching."""

    file_path: str
    co_accessed_with: Counter = field(default_factory=Counter)
    access_count: int = 0


class WorkflowLearner:
    """Learns from your workflow and adapts routing, caching, and worker selection.

    Persisted to ~/.coolcode/learnings.json.
    """

    def __init__(self, persist_path: str | None = None):
        self._patterns: dict[str, TaskPattern] = {}
        self._file_patterns: dict[str, FileAccessPattern] = {}
        self._task_history: list[dict[str, Any]] = []
        self._persist_path = Path(persist_path) if persist_path else None

        if self._persist_path and self._persist_path.exists():
            self._load()

    def _load(self) -> None:
        try:
            data = json.loads(self._persist_path.read_text())
            for key, p in data.get("patterns", {}).items():
                self._patterns[key] = TaskPattern(**p)
            self._task_history = data.get("history", [])[-500:]  # keep last 500
        except (json.JSONDecodeError, KeyError, TypeError):
            logger.warning("Failed to load learnings, starting fresh")

    def save(self) -> None:
        if not self._persist_path:
            return
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "patterns": {
                k: {
                    "keywords": p.keywords,
                    "best_worker_type": p.best_worker_type,
                    "avg_confidence": p.avg_confidence,
                    "avg_duration_ms": p.avg_duration_ms,
                    "success_count": p.success_count,
                    "fail_count": p.fail_count,
                    "last_used": p.last_used,
                }
                for k, p in self._patterns.items()
            },
            "history": self._task_history[-500:],
        }
        self._persist_path.write_text(json.dumps(data, indent=2))

    def record_execution(
        self,
        task: str,
        worker_type: str,
        success: bool,
        confidence: float,
        duration_ms: float,
        files_accessed: list[str] | None = None,
    ) -> None:
        """Record the outcome of a task execution for future learning."""
        keywords = self._extract_keywords(task)
        pattern_key = f"{worker_type}:{','.join(sorted(keywords[:5]))}"

        if pattern_key in self._patterns:
            p = self._patterns[pattern_key]
            if success:
                p.success_count += 1
            else:
                p.fail_count += 1
            # Running average
            total = p.success_count + p.fail_count
            p.avg_confidence = (p.avg_confidence * (total - 1) + confidence) / total
            p.avg_duration_ms = (p.avg_duration_ms * (total - 1) + duration_ms) / total
            p.last_used = time.time()
        else:
            self._patterns[pattern_key] = TaskPattern(
                keywords=keywords[:5],
                best_worker_type=worker_type,
                avg_confidence=confidence,
                avg_duration_ms=duration_ms,
                success_count=1 if success else 0,
                fail_count=0 if success else 1,
            )

        # Track file co-access patterns
        if files_accessed and len(files_accessed) > 1:
            for f in files_accessed:
                if f not in self._file_patterns:
                    self._file_patterns[f] = FileAccessPattern(file_path=f)
                self._file_patterns[f].access_count += 1
                for other in files_accessed:
                    if other != f:
                        self._file_patterns[f].co_accessed_with[other] += 1

        # Append to history
        self._task_history.append({
            "task": task[:200],
            "worker_type": worker_type,
            "success": success,
            "confidence": confidence,
            "duration_ms": duration_ms,
            "timestamp": time.time(),
        })

    def suggest_worker_count(self, task: str) -> int:
        """Suggest optimal worker count based on learned task complexity."""
        keywords = set(self._extract_keywords(task))
        matching_patterns = []

        for p in self._patterns.values():
            overlap = keywords & set(p.keywords)
            if overlap:
                matching_patterns.append(p)

        if not matching_patterns:
            return 1  # default: single worker for unknown tasks

        avg_confidence = sum(p.avg_confidence for p in matching_patterns) / len(matching_patterns)

        # Low confidence tasks benefit from more workers (racing)
        if avg_confidence < 0.5:
            return 3
        elif avg_confidence < 0.7:
            return 2
        else:
            return 1  # high confidence = one worker is enough, save tokens

    def suggest_skip_workers(self, task: str) -> list[str]:
        """Suggest worker types to skip based on past failures."""
        keywords = set(self._extract_keywords(task))
        skip = []

        for p in self._patterns.values():
            overlap = keywords & set(p.keywords)
            if overlap and p.success_rate < 0.3 and (p.success_count + p.fail_count) >= 3:
                skip.append(p.best_worker_type)

        return skip

    def get_related_files(self, file_path: str, top_k: int = 5) -> list[str]:
        """Get files frequently accessed together with the given file."""
        if file_path not in self._file_patterns:
            return []
        pattern = self._file_patterns[file_path]
        return [f for f, _ in pattern.co_accessed_with.most_common(top_k)]

    @property
    def stats(self) -> dict[str, Any]:
        patterns = list(self._patterns.values())
        if not patterns:
            return {"patterns_learned": 0, "total_executions": 0}
        return {
            "patterns_learned": len(patterns),
            "total_executions": len(self._task_history),
            "avg_success_rate": f"{sum(p.success_rate for p in patterns) / len(patterns):.0%}",
            "top_patterns": [
                {"key": k, "score": f"{p.score:.2f}", "uses": p.success_count + p.fail_count}
                for k, p in sorted(self._patterns.items(), key=lambda x: x[1].score, reverse=True)[:5]
            ],
        }

    @staticmethod
    def _extract_keywords(task: str) -> list[str]:
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "can", "shall", "to", "of", "in", "for",
            "on", "with", "at", "by", "from", "as", "into", "and", "but", "or",
            "not", "no", "this", "that", "it", "its", "i", "me", "my", "we",
            "you", "your", "he", "she", "they", "them", "please", "tell",
        }
        words = task.lower().split()
        return [
            w.strip(".,!?;:\"'()[]{}") for w in words
            if w.strip(".,!?;:\"'()[]{}") not in stop_words and len(w) > 2
        ]
