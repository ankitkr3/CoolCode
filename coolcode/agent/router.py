"""Intelligent task routing with learned patterns."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from coolcode.agent.worker import WorkerType

logger = logging.getLogger(__name__)


@dataclass
class RoutingRecord:
    """A record of a past routing decision and its outcome."""

    task_keywords: list[str]
    routed_to: list[str]  # WorkerType values
    success: bool
    quality_score: float  # 0.0-1.0
    timestamp: float = field(default_factory=time.time)


class TaskRouter:
    """Learns from past routing decisions to improve future task assignment.

    Maintains a history of task → worker type mappings with outcomes,
    and uses pattern matching to route new tasks to the most
    successful worker types.
    """

    def __init__(self, persist_path: str | None = None):
        self._history: list[RoutingRecord] = []
        self._persist_path = Path(persist_path) if persist_path else None
        if self._persist_path and self._persist_path.exists():
            self._load()

    def _load(self) -> None:
        data = json.loads(self._persist_path.read_text())
        self._history = [RoutingRecord(**r) for r in data]

    def save(self) -> None:
        if not self._persist_path:
            return
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        data = [
            {
                "task_keywords": r.task_keywords,
                "routed_to": r.routed_to,
                "success": r.success,
                "quality_score": r.quality_score,
                "timestamp": r.timestamp,
            }
            for r in self._history
        ]
        self._persist_path.write_text(json.dumps(data, indent=2))

    def record_outcome(
        self,
        task: str,
        routed_to: list[WorkerType],
        success: bool,
        quality_score: float,
    ) -> None:
        """Record the outcome of a routing decision for future learning."""
        keywords = self._extract_keywords(task)
        record = RoutingRecord(
            task_keywords=keywords,
            routed_to=[w.value for w in routed_to],
            success=success,
            quality_score=quality_score,
        )
        self._history.append(record)

    def suggest_workers(self, task: str, top_k: int = 3) -> list[tuple[WorkerType, float]]:
        """Suggest worker types based on learned patterns.

        Returns list of (WorkerType, confidence) sorted by confidence descending.
        """
        if not self._history:
            return []

        task_keywords = set(self._extract_keywords(task))
        if not task_keywords:
            return []

        # Score each worker type based on keyword overlap with successful past routings
        worker_scores: dict[str, list[float]] = {}

        for record in self._history:
            overlap = len(task_keywords & set(record.task_keywords))
            if overlap == 0:
                continue
            relevance = overlap / max(len(task_keywords), len(record.task_keywords))
            score = relevance * record.quality_score if record.success else relevance * -0.5
            for wt in record.routed_to:
                worker_scores.setdefault(wt, []).append(score)

        # Average scores per worker type
        avg_scores: list[tuple[WorkerType, float]] = []
        for wt_str, scores in worker_scores.items():
            avg = sum(scores) / len(scores)
            try:
                avg_scores.append((WorkerType(wt_str), max(0.0, min(1.0, avg))))
            except ValueError:
                continue

        avg_scores.sort(key=lambda x: x[1], reverse=True)
        return avg_scores[:top_k]

    @staticmethod
    def _extract_keywords(task: str) -> list[str]:
        """Extract meaningful keywords from a task description."""
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "can", "shall", "to", "of", "in", "for",
            "on", "with", "at", "by", "from", "as", "into", "through", "during",
            "before", "after", "above", "below", "between", "and", "but", "or",
            "not", "no", "this", "that", "these", "those", "it", "its", "i", "me",
            "my", "we", "our", "you", "your", "he", "she", "they", "them", "please",
        }
        words = task.lower().split()
        return [w.strip(".,!?;:\"'()[]{}") for w in words if w.strip(".,!?;:\"'()[]{}") not in stop_words and len(w) > 2]
