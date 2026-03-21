"""Cost tracking and budget management for Cool Code.

Tracks token usage per provider per task, calculates costs,
enforces budget caps, and persists daily cost logs.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Cost per 1M tokens (input, output) — mirrors provider.py's COST_TABLE
COST_TABLE: dict[str, tuple[float, float]] = {
    "claude-opus-4-6": (15.0, 75.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5-20251001": (0.80, 4.0),
    "MiniMax-M2.7-highspeed": (0.5, 2.0),
    "MiniMax-M2.7": (1.0, 4.0),
    "MiniMax-M2.5": (0.15, 1.1),
    "MiniMax-M2.5-highspeed": (0.3, 2.2),
}


@dataclass
class UsageRecord:
    """A single token usage record."""

    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    timestamp: float = field(default_factory=time.time)
    task_snippet: str = ""


class CostTracker:
    """Tracks token costs across providers for a session.

    Thread-safe — workers recording costs from parallel threads is safe.
    Persists daily costs to ~/.coolcode/costs.jsonl.
    """

    def __init__(self, persist_dir: str | None = None):
        self._records: list[UsageRecord] = []
        self._lock = threading.Lock()
        self._persist_dir = Path(persist_dir or Path.home() / ".coolcode")
        self._persist_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost in USD for a given model and token counts."""
        cost_per_m = COST_TABLE.get(model, (5.0, 25.0))  # fallback to moderate cost
        input_cost = (input_tokens / 1_000_000) * cost_per_m[0]
        output_cost = (output_tokens / 1_000_000) * cost_per_m[1]
        return input_cost + output_cost

    def record(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        task_snippet: str = "",
    ) -> float:
        """Record token usage and return the cost in USD."""
        cost = self.calculate_cost(model, input_tokens, output_tokens)
        record = UsageRecord(
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            task_snippet=task_snippet[:100],
        )
        with self._lock:
            self._records.append(record)
        self._persist_record(record)
        return cost

    @property
    def session_total(self) -> float:
        """Total cost for the current session."""
        with self._lock:
            return sum(r.cost_usd for r in self._records)

    @property
    def session_tokens(self) -> tuple[int, int]:
        """Total (input_tokens, output_tokens) for the session."""
        with self._lock:
            inp = sum(r.input_tokens for r in self._records)
            out = sum(r.output_tokens for r in self._records)
            return inp, out

    def provider_breakdown(self) -> dict[str, dict[str, Any]]:
        """Cost breakdown by provider."""
        breakdown: dict[str, dict[str, Any]] = {}
        with self._lock:
            for r in self._records:
                key = f"{r.provider}:{r.model}"
                if key not in breakdown:
                    breakdown[key] = {
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "cost_usd": 0.0,
                        "calls": 0,
                    }
                breakdown[key]["input_tokens"] += r.input_tokens
                breakdown[key]["output_tokens"] += r.output_tokens
                breakdown[key]["cost_usd"] += r.cost_usd
                breakdown[key]["calls"] += 1
        return breakdown

    def task_cost(self, last_n: int = 1) -> float:
        """Cost of the last N recorded tasks."""
        with self._lock:
            return sum(r.cost_usd for r in self._records[-last_n:])

    def daily_total(self) -> float:
        """Total cost for today (from persisted log)."""
        log_path = self._persist_dir / "costs.jsonl"
        if not log_path.exists():
            return self.session_total
        today = date.today().isoformat()
        total = 0.0
        try:
            for line in log_path.read_text().splitlines():
                if not line.strip():
                    continue
                entry = json.loads(line)
                if entry.get("date") == today:
                    total += entry.get("cost_usd", 0.0)
        except (json.JSONDecodeError, IOError):
            pass
        return total

    def check_budget(self, daily_limit: float = 0.0, session_limit: float = 0.0) -> BudgetStatus:
        """Check if spending is within budget.

        Returns BudgetStatus with ok/warning/exceeded state.
        Limits of 0.0 mean unlimited.
        """
        session_cost = self.session_total
        daily_cost = self.daily_total()

        if session_limit > 0 and session_cost >= session_limit:
            return BudgetStatus("exceeded", "session", session_cost, session_limit)
        if daily_limit > 0 and daily_cost >= daily_limit:
            return BudgetStatus("exceeded", "daily", daily_cost, daily_limit)

        # Warning at 80%
        if session_limit > 0 and session_cost >= session_limit * 0.8:
            return BudgetStatus("warning", "session", session_cost, session_limit)
        if daily_limit > 0 and daily_cost >= daily_limit * 0.8:
            return BudgetStatus("warning", "daily", daily_cost, daily_limit)

        return BudgetStatus("ok", "", session_cost, 0.0)

    def _persist_record(self, record: UsageRecord) -> None:
        """Append a cost record to the daily log file."""
        log_path = self._persist_dir / "costs.jsonl"
        entry = {
            "date": date.today().isoformat(),
            "provider": record.provider,
            "model": record.model,
            "input_tokens": record.input_tokens,
            "output_tokens": record.output_tokens,
            "cost_usd": record.cost_usd,
            "timestamp": record.timestamp,
            "task": record.task_snippet,
        }
        try:
            with open(log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except IOError as e:
            logger.warning(f"Failed to persist cost record: {e}")

    @property
    def records(self) -> list[UsageRecord]:
        with self._lock:
            return list(self._records)


@dataclass
class BudgetStatus:
    """Result of a budget check."""

    status: str  # "ok", "warning", "exceeded"
    scope: str  # "session", "daily", or ""
    current_cost: float
    limit: float

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    @property
    def exceeded(self) -> bool:
        return self.status == "exceeded"

    def __str__(self) -> str:
        if self.status == "ok":
            return f"Budget OK (${self.current_cost:.4f})"
        pct = (self.current_cost / self.limit * 100) if self.limit > 0 else 0
        return f"Budget {self.status} ({self.scope}): ${self.current_cost:.4f} / ${self.limit:.2f} ({pct:.0f}%)"
