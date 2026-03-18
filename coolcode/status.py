"""Live status updates and fun Hindi status messages."""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field

# Fun Hindi status messages — rotated during work
HINDI_STATUS_MESSAGES: list[str] = [
    "kaam chal rha hai...",
    "soch rha hoon...",
    "code padh rha hoon...",
    "dimag laga rha hoon...",
    "thoda ruko, kuch bana rha hoon...",
    "jugaad lag rha hai...",
    "ek minute, samajh aa rha hai...",
    "chal rha hai bhai, tension mat le...",
    "abhi solve karta hoon...",
    "deep thinking mode on hai...",
    "code ko ghoor rha hoon...",
    "pura plan bana rha hoon...",
    "tera kaam ho jayega, chill kar...",
    "bugs dhundh rha hoon...",
    "logic bitha rha hoon...",
    "haan haan, karta hoon...",
    "thoda sa aur...",
    "almost ho gya...",
    "full focus mode...",
    "code mein ghusa hua hoon...",
    "pattern samajh aa gya...",
    "solution mil gya lagta hai...",
    "bas ek aur step...",
    "chinta mat kar, sab hoga...",
    "mehnat rang la rhi hai...",
    "ek aur file padh leta hoon...",
    "pehle samjho, phir karo...",
    "full power laga rha hoon...",
    "swarm ko bola hai kaam karo...",
    "saare agents lage hue hain...",
]


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
    """

    def __init__(self) -> None:
        self._updates: asyncio.Queue[StatusUpdate] = asyncio.Queue()
        self._history: list[StatusUpdate] = []
        self._hindi_index = 0
        random.shuffle(HINDI_STATUS_MESSAGES)

    def emit(self, source: str, action: str, detail: str = "") -> None:
        """Emit a status update (non-blocking)."""
        update = StatusUpdate(source=source, action=action, detail=detail)
        self._history.append(update)
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
        """Get the next fun Hindi message (cycles through all)."""
        msg = HINDI_STATUS_MESSAGES[self._hindi_index % len(HINDI_STATUS_MESSAGES)]
        self._hindi_index += 1
        return msg

    @property
    def history(self) -> list[StatusUpdate]:
        return list(self._history)
