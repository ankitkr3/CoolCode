"""Token Optimizer — reduce API costs via context compression, result caching, and smart truncation.

Integrates into the agentic flow to minimize token usage without losing quality.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ResultCache:
    """LRU cache for LLM results — don't re-ask questions we already answered.

    Keyed by a hash of (task + worker_type + relevant_context).
    Persisted to disk so cache survives across sessions.
    """

    def __init__(self, cache_dir: str | None = None, max_size: int = 500, ttl_hours: int = 24):
        self._max_size = max_size
        self._ttl_seconds = ttl_hours * 3600
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._cache_path = Path(cache_dir) / "result_cache.json" if cache_dir else None
        self._hits = 0
        self._misses = 0
        if self._cache_path and self._cache_path.exists():
            self._load()

    def _load(self) -> None:
        try:
            data = json.loads(self._cache_path.read_text())
            now = time.time()
            for key, entry in data.items():
                if now - entry.get("timestamp", 0) < self._ttl_seconds:
                    self._cache[key] = entry
        except (json.JSONDecodeError, KeyError):
            pass

    def save(self) -> None:
        if not self._cache_path:
            return
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_path.write_text(json.dumps(dict(self._cache), indent=2))

    @staticmethod
    def _make_key(task: str, worker_type: str, context: str = "") -> str:
        raw = f"{task.strip().lower()}|{worker_type}|{context[:200]}"
        return hashlib.sha256(raw.encode()).hexdigest()[:32]

    def get(self, task: str, worker_type: str, context: str = "") -> str | None:
        """Get cached result. Returns None on miss."""
        key = self._make_key(task, worker_type, context)
        if key in self._cache:
            entry = self._cache[key]
            if time.time() - entry["timestamp"] < self._ttl_seconds:
                self._cache.move_to_end(key)
                self._hits += 1
                logger.info(f"Cache HIT for {worker_type}: {task[:50]}")
                return entry["result"]
            else:
                del self._cache[key]
        self._misses += 1
        return None

    def put(self, task: str, worker_type: str, result: str, context: str = "") -> None:
        """Cache a result."""
        key = self._make_key(task, worker_type, context)
        self._cache[key] = {
            "task": task[:200],
            "worker_type": worker_type,
            "result": result,
            "timestamp": time.time(),
        }
        self._cache.move_to_end(key)
        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

    @property
    def stats(self) -> dict[str, Any]:
        total = self._hits + self._misses
        return {
            "size": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{self._hits / total:.0%}" if total > 0 else "N/A",
        }


class ContextCompressor:
    """Compress context sent to LLMs to reduce token usage.

    Strategies:
    1. File truncation — only send first/last N lines of large files
    2. Import-only mode — for architecture questions, only send imports + class/function signatures
    3. Deduplication — don't repeat context already in the system prompt
    4. Smart summarization — replace large tool outputs with summaries
    """

    def __init__(self, max_context_chars: int = 30000):
        self.max_context_chars = max_context_chars

    def compress_file_content(self, content: str, file_path: str = "", mode: str = "smart") -> str:
        """Compress file content based on mode.

        Modes:
        - "full": no compression (small files)
        - "truncate": first 50 + last 20 lines
        - "signatures": only imports, class defs, function defs
        - "smart": auto-pick based on file size
        """
        lines = content.splitlines()

        if mode == "smart":
            if len(lines) <= 80:
                mode = "full"
            elif len(lines) <= 300:
                mode = "truncate"
            else:
                mode = "signatures"

        if mode == "full":
            return content

        if mode == "truncate":
            head = lines[:50]
            tail = lines[-20:]
            skipped = len(lines) - 70
            return "\n".join(head) + f"\n\n... ({skipped} lines skipped) ...\n\n" + "\n".join(tail)

        if mode == "signatures":
            sig_lines = []
            for i, line in enumerate(lines):
                stripped = line.strip()
                if (
                    stripped.startswith(("import ", "from ", "require(", "require "))
                    or stripped.startswith(("class ", "def ", "async def ", "function ", "export "))
                    or stripped.startswith(("struct ", "impl ", "trait ", "enum ", "type ", "interface "))
                    or stripped.startswith(("const ", "let ", "var "))
                    and "=" in stripped
                    and "function" in stripped
                ):
                    sig_lines.append(f"{i + 1}: {line}")
            header = f"# {file_path} ({len(lines)} lines) — signatures only"
            return header + "\n" + "\n".join(sig_lines) if sig_lines else header + "\n(no signatures found)"

        return content

    def compress_tool_output(self, output: str, tool_name: str = "") -> str:
        """Compress tool output if it's too large."""
        if len(output) <= 5000:
            return output

        lines = output.splitlines()

        if tool_name in ("grep_search", "glob_search"):
            # For search results: keep first 30 results
            if len(lines) > 30:
                return "\n".join(lines[:30]) + f"\n... ({len(lines) - 30} more results)"

        if tool_name in ("list_dir",):
            # For directory listings: keep all (usually short)
            return output

        if tool_name in ("execute_shell",):
            # For shell: keep first 50 + last 20 lines
            if len(lines) > 70:
                head = lines[:50]
                tail = lines[-20:]
                return "\n".join(head) + f"\n... ({len(lines) - 70} lines skipped)\n" + "\n".join(tail)

        # Generic: truncate at max chars
        if len(output) > self.max_context_chars:
            return output[: self.max_context_chars] + f"\n... (truncated from {len(output)} chars)"

        return output

    def estimate_tokens(self, text: str) -> int:
        """Rough token estimate (1 token ≈ 4 chars for English/code)."""
        return len(text) // 4


class TokenOptimizer:
    """Main optimizer that combines caching and compression.

    Integrates into the swarm flow:
    1. Before spawning workers → check cache for identical/similar tasks
    2. Before passing context → compress files and tool outputs
    3. After getting results → cache them for future use
    4. Track savings → report how many tokens/dollars saved
    """

    def __init__(self, cache_dir: str | None = None):
        self.cache = ResultCache(cache_dir=cache_dir)
        self.compressor = ContextCompressor()
        self._tokens_saved = 0
        self._tokens_used = 0
        self._cache_saves = 0

    def check_cache(self, task: str, worker_type: str) -> str | None:
        """Check if we already have a cached result for this task."""
        return self.cache.get(task, worker_type)

    def cache_result(self, task: str, worker_type: str, result: str) -> None:
        """Cache a successful result."""
        self.cache.put(task, worker_type, result)
        self._cache_saves += 1

    def compress_for_worker(self, task: str, file_contents: dict[str, str]) -> dict[str, str]:
        """Compress file contents before sending to a worker."""
        compressed = {}
        for path, content in file_contents.items():
            original_tokens = self.compressor.estimate_tokens(content)
            comp = self.compressor.compress_file_content(content, file_path=path)
            new_tokens = self.compressor.estimate_tokens(comp)
            self._tokens_saved += original_tokens - new_tokens
            self._tokens_used += new_tokens
            compressed[path] = comp
        return compressed

    def save(self) -> None:
        self.cache.save()

    @property
    def stats(self) -> dict[str, Any]:
        total_tokens = self._tokens_used + self._tokens_saved
        return {
            "tokens_used": self._tokens_used,
            "tokens_saved": self._tokens_saved,
            "savings_pct": f"{self._tokens_saved / total_tokens:.0%}" if total_tokens > 0 else "N/A",
            "cache_saves": self._cache_saves,
            "cache": self.cache.stats,
        }
