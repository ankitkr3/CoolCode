"""Live status updates and context-aware Hindi status messages."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Context-aware Hindi messages — matched to what's actually happening
HINDI_MESSAGES: dict[str, list[str]] = {
    # Phase: analyzing / routing (before workers start)
    "analyzing": [
        "soch rha hoon...",
        "dimag laga rha hoon...",
        "samajh rha hoon kya karna hai...",
        "pehle samjho, phir karo...",
        "deep thinking mode on hai...",
    ],
    # Phase: reading files / researching
    "reading": [
        "code padh rha hoon...",
        "code ko ghoor rha hoon...",
        "ek aur file padh leta hoon...",
        "code mein ghusa hua hoon...",
        "pattern samajh aa gya...",
    ],
    # Phase: workers executing / racing
    "executing": [
        "kaam chal rha hai...",
        "swarm ko bola hai kaam karo...",
        "saare agents lage hue hain...",
        "full power laga rha hoon...",
        "full focus mode...",
        "jugaad lag rha hai...",
        "chal rha hai bhai, tension mat le...",
        "tera kaam ho jayega, chill kar...",
        "haan haan, karta hoon...",
    ],
    # Phase: writing / building / coding
    "writing": [
        "thoda ruko, kuch bana rha hoon...",
        "abhi solve karta hoon...",
        "logic bitha rha hoon...",
        "pura plan bana rha hoon...",
    ],
    # Phase: debugging / fixing
    "debugging": [
        "bugs dhundh rha hoon...",
        "kya galat hai dekh rha hoon...",
        "root cause pakda...",
    ],
    # Phase: security / review
    "security": [
        "security check chal rha hai...",
        "vulnerabilities dhundh rha hoon...",
        "attack surface map kar rha hoon...",
        "OWASP checklist chala rha hoon...",
    ],
    # Phase: finishing up
    "finishing": [
        "thoda sa aur...",
        "almost ho gya...",
        "solution mil gya lagta hai...",
        "bas ek aur step...",
        "mehnat rang la rhi hai...",
        "chinta mat kar, sab hoga...",
    ],
}

# Flat fallback — used when phase can't be determined
HINDI_FALLBACK = [msg for msgs in HINDI_MESSAGES.values() for msg in msgs]


@dataclass
class StatusUpdate:
    """A single status update from a worker or the swarm."""

    source: str  # worker_id or "swarm" or "queen"
    action: str  # what's happening: "spawning", "thinking", "reading", "writing", etc.
    detail: str = ""  # extra info: file name, function, etc.
    timestamp: float = field(default_factory=time.time)


# Map tool actions to human-readable verbs (port of Claude Code sessionRunner.ts:69-105)
TOOL_VERBS: dict[str, str] = {
    "reading": "Reading",
    "read": "Read",
    "read (cached)": "Read (cached)",
    "writing": "Writing",
    "wrote": "Wrote",
    "editing": "Editing",
    "edited": "Edited",
    "listing": "Listing",
    "listed": "Listed",
    "searching": "Searching",
    "found": "Found",
    "running": "Running",
    "ran": "Ran",
    "grepping": "Grepping",
    "grep done": "Grep done",
    "finding": "Finding",
    "found def": "Found def",
    "git": "Git",
    "started": "Started",
    "thinking": "Thinking",
    "streaming": "Streaming",
    "done": "Done",
    "failed": "Failed",
    "timeout": "Timed out",
    "timeout_partial": "Timed out (partial)",
    "cancelling": "Cancelling",
    "cancelled": "Cancelled",
}


def verb_for(action: str) -> str:
    """Return a human-readable verb for a tool action."""
    return TOOL_VERBS.get(action, action)


@dataclass
class WorkerSnapshot:
    """Live snapshot of a single worker's state."""

    worker_id: str
    state: str = "pending"  # pending, running, done, failed, timeout, cancelled
    action: str = ""  # current action verb (Reading, Writing, ...)
    detail: str = ""  # current detail (file path, command, ...)
    started_at: float = 0.0
    updated_at: float = field(default_factory=time.time)
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    chunks: int = 0  # streaming chunks processed

    @property
    def elapsed_ms(self) -> float:
        if not self.started_at:
            return 0.0
        end = time.time() if self.state in ("pending", "running") else self.updated_at
        return (end - self.started_at) * 1000

    @property
    def is_terminal(self) -> bool:
        return self.state in ("done", "failed", "timeout", "cancelled")


class StatusTracker:
    """Collects real-time status updates from workers and the swarm.

    The CLI reads from this to show live progress.
    Hindi messages are context-aware — they match the current execution phase.
    """

    # Map (source, action) keywords to message phases
    _PHASE_RULES: list[tuple[list[str], str]] = [
        (["analyzing", "routing", "deciding", "samajh"], "analyzing"),
        (["reading", "read", "listing", "listed", "grepping", "grep", "finding", "searching"], "reading"),
        (["writing", "editing", "wrote", "edited", "committing"], "writing"),
        (["debug", "diagnos", "fix"], "debugging"),
        (["security", "cyber", "owasp", "cwe", "vuln", "injection", "auth"], "security"),
        (["spawning", "racing", "started", "thinking", "executing", "working"], "executing"),
        (["evaluating", "consensus", "merging", "done", "complete", "cache"], "finishing"),
    ]

    def __init__(self, transcript_path: str | None = None) -> None:
        self._updates: asyncio.Queue[StatusUpdate] = asyncio.Queue()
        self._history: list[StatusUpdate] = []
        self._ring: list[StatusUpdate] = []  # bounded ring buffer (last 10)
        self._ring_max = 10
        self._workers: dict[str, WorkerSnapshot] = {}
        self._phase = "analyzing"  # current phase for Hindi messages
        self._phase_index: dict[str, int] = {k: 0 for k in HINDI_MESSAGES}
        # Shuffle each phase's messages independently
        for msgs in HINDI_MESSAGES.values():
            random.shuffle(msgs)

        # Transcript log — append-only NDJSON for post-hoc debugging
        # Port of Claude Code sessionRunner.ts transcript pattern
        self._transcript_path: Path | None = None
        if transcript_path:
            try:
                p = Path(transcript_path)
                p.parent.mkdir(parents=True, exist_ok=True)
                self._transcript_path = p
            except Exception as e:
                logger.debug(f"Failed to open transcript {transcript_path}: {e}")

    def _detect_phase(self, source: str, action: str, detail: str) -> str:
        """Detect the current execution phase from the latest status update."""
        combined = f"{source} {action} {detail}".lower()
        for keywords, phase in self._PHASE_RULES:
            if any(kw in combined for kw in keywords):
                return phase
        return self._phase  # keep current phase if no match

    def emit(self, source: str, action: str, detail: str = "") -> None:
        """Emit a status update (non-blocking). Also updates the current phase."""
        update = StatusUpdate(source=source, action=action, detail=detail)
        self._history.append(update)
        # Bounded ring buffer (port of Claude Code sessionRunner.ts ring buffer)
        self._ring.append(update)
        if len(self._ring) > self._ring_max:
            self._ring.pop(0)
        self._phase = self._detect_phase(source, action, detail)
        self._update_worker_state(source, action, detail, update.timestamp)
        self._append_transcript(update)
        try:
            self._updates.put_nowait(update)
        except asyncio.QueueFull:
            pass  # Drop if queue is full — UI will catch up

    def _append_transcript(self, update: StatusUpdate) -> None:
        """Append a status update to the NDJSON transcript log (fire-and-forget)."""
        if self._transcript_path is None:
            return
        try:
            line = json.dumps({
                "ts": update.timestamp,
                "source": update.source,
                "action": update.action,
                "detail": update.detail,
            }) + "\n"
            with self._transcript_path.open("a", encoding="utf-8") as f:
                f.write(line)
        except Exception as e:
            logger.debug(f"Transcript write failed: {e}")

    def _update_worker_state(
        self, source: str, action: str, detail: str, ts: float
    ) -> None:
        """Maintain per-worker snapshots based on emitted events."""
        # Only track real workers — worker IDs contain a dash and a number
        # Skip: swarm, queen, router, goal, user, learner, auto
        if source in ("swarm", "queen", "router", "goal", "user", "learner", "auto"):
            return

        snap = self._workers.get(source)
        if snap is None:
            snap = WorkerSnapshot(worker_id=source, started_at=ts)
            self._workers[source] = snap

        snap.updated_at = ts

        # Parse token info from cost messages: "$0.0042 (123 in / 456 out)"
        if action == "cost" and "in /" in detail:
            try:
                import re
                m = re.search(r"\$([\d.]+)\s*\((\d+)\s*in\s*/\s*(\d+)\s*out\)", detail)
                if m:
                    snap.cost_usd = float(m.group(1))
                    snap.tokens_in = int(m.group(2))
                    snap.tokens_out = int(m.group(3))
            except Exception:
                pass
            return

        # Parse chunk count from streaming: "chunk 9 (5 messages)"
        if action == "streaming" and "chunk" in detail:
            try:
                import re
                m = re.search(r"chunk\s+(\d+)", detail)
                if m:
                    snap.chunks = int(m.group(1))
            except Exception:
                pass

        # Terminal states
        if action in ("done", "failed", "timeout", "timeout_partial", "cancelled"):
            snap.state = {
                "done": "done",
                "failed": "failed",
                "timeout": "timeout",
                "timeout_partial": "timeout",
                "cancelled": "cancelled",
            }[action]
            snap.action = verb_for(action)
            snap.detail = detail
            return

        # Running state
        snap.state = "running"
        snap.action = verb_for(action)
        snap.detail = detail[:60] if detail else ""

    def worker_snapshots(self) -> list[WorkerSnapshot]:
        """Return a stable-sorted list of current worker snapshots."""
        return sorted(
            self._workers.values(),
            key=lambda w: (w.is_terminal, w.worker_id),
        )

    def ring(self) -> list[StatusUpdate]:
        """Return the bounded ring buffer (latest 10 events)."""
        return list(self._ring)

    async def get(self, timeout: float = 0.5) -> StatusUpdate | None:
        """Get the next status update, or None on timeout."""
        try:
            return await asyncio.wait_for(self._updates.get(), timeout=timeout)
        except (asyncio.TimeoutError, TimeoutError):
            return None

    def next_hindi(self) -> str:
        """Get the next Hindi message matching the current execution phase."""
        msgs = HINDI_MESSAGES.get(self._phase, HINDI_FALLBACK)
        idx = self._phase_index.get(self._phase, 0)
        msg = msgs[idx % len(msgs)]
        self._phase_index[self._phase] = idx + 1
        return msg

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def history(self) -> list[StatusUpdate]:
        return list(self._history)
