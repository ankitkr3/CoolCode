"""Live status updates and context-aware Hindi status messages."""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field

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

    def __init__(self) -> None:
        self._updates: asyncio.Queue[StatusUpdate] = asyncio.Queue()
        self._history: list[StatusUpdate] = []
        self._phase = "analyzing"  # current phase for Hindi messages
        self._phase_index: dict[str, int] = {k: 0 for k in HINDI_MESSAGES}
        # Shuffle each phase's messages independently
        for msgs in HINDI_MESSAGES.values():
            random.shuffle(msgs)

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
        self._phase = self._detect_phase(source, action, detail)
        try:
            self._updates.put_nowait(update)
        except asyncio.QueueFull:
            pass  # Drop if queue is full — UI will catch up

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
