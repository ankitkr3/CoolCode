"""Collective memory — shared knowledge base with LRU cache and SQLite persistence."""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from collections import OrderedDict
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class MemoryType(str, Enum):
    """The 8 types of collective memory."""

    FACT = "fact"  # Verified facts about the codebase
    PATTERN = "pattern"  # Recurring code patterns
    INSIGHT = "insight"  # Agent-derived insights
    DECISION = "decision"  # Architecture/design decisions
    ERROR = "error"  # Known errors and fixes
    CONTEXT = "context"  # Conversation/task context
    PREFERENCE = "preference"  # User preferences
    LEARNING = "learning"  # Self-learned behaviors


class CollectiveMemory:
    """Shared memory accessible by all agents in the swarm.

    Uses SQLite for persistence and an LRU cache for fast access.
    Supports 8 memory types, tagging, and relevance scoring.
    """

    def __init__(self, db_path: str, lru_size: int = 1024):
        self._db_path = db_path
        self._lru_size = lru_size
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    memory_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT DEFAULT '{}',
                    tags TEXT DEFAULT '[]',
                    relevance_score REAL DEFAULT 1.0,
                    access_count INTEGER DEFAULT 0,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_relevance ON memories(relevance_score DESC)
            """)

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    def _to_cache(self, key: str, value: dict[str, Any]) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        if len(self._cache) > self._lru_size:
            self._cache.popitem(last=False)

    def store(
        self,
        memory_id: str,
        memory_type: MemoryType,
        content: str,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        relevance_score: float = 1.0,
    ) -> None:
        """Store or update a memory."""
        now = time.time()
        row = {
            "id": memory_id,
            "memory_type": memory_type.value,
            "content": content,
            "metadata": json.dumps(metadata or {}),
            "tags": json.dumps(tags or []),
            "relevance_score": relevance_score,
            "created_at": now,
            "updated_at": now,
        }
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO memories (id, memory_type, content, metadata, tags, relevance_score, created_at, updated_at)
                VALUES (:id, :memory_type, :content, :metadata, :tags, :relevance_score, :created_at, :updated_at)
                ON CONFLICT(id) DO UPDATE SET
                    content = :content,
                    metadata = :metadata,
                    tags = :tags,
                    relevance_score = :relevance_score,
                    updated_at = :updated_at
                """,
                row,
            )
        self._to_cache(memory_id, {**row, "access_count": 0})

    def recall(self, memory_id: str) -> dict[str, Any] | None:
        """Retrieve a specific memory by ID, with LRU caching."""
        if memory_id in self._cache:
            self._cache.move_to_end(memory_id)
            return self._cache[memory_id]
        with self._conn() as conn:
            conn.execute(
                "UPDATE memories SET access_count = access_count + 1 WHERE id = ?",
                (memory_id,),
            )
            row = conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        if row is None:
            return None
        result = self._row_to_dict(row)
        self._to_cache(memory_id, result)
        return result

    def search(
        self,
        memory_type: MemoryType | None = None,
        tags: list[str] | None = None,
        limit: int = 20,
        min_relevance: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Search memories by type, tags, or relevance threshold."""
        query = "SELECT * FROM memories WHERE relevance_score >= ?"
        params: list[Any] = [min_relevance]
        if memory_type:
            query += " AND memory_type = ?"
            params.append(memory_type.value)
        query += " ORDER BY relevance_score DESC, updated_at DESC LIMIT ?"
        params.append(limit)
        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
        results = [self._row_to_dict(r) for r in rows]
        if tags:
            tag_set = set(tags)
            results = [r for r in results if tag_set & set(json.loads(r.get("tags", "[]")))]
        return results

    def decay_relevance(self, factor: float = 0.95) -> None:
        """Apply time-based decay to all relevance scores."""
        with self._conn() as conn:
            conn.execute("UPDATE memories SET relevance_score = relevance_score * ?", (factor,))
        self._cache.clear()

    def prune(self, min_relevance: float = 0.01) -> int:
        """Remove memories below the relevance threshold. Returns count removed."""
        with self._conn() as conn:
            cursor = conn.execute(
                "DELETE FROM memories WHERE relevance_score < ?", (min_relevance,)
            )
            removed = cursor.rowcount
        self._cache.clear()
        return removed

    @staticmethod
    def _row_to_dict(row: tuple) -> dict[str, Any]:
        keys = [
            "id", "memory_type", "content", "metadata", "tags",
            "relevance_score", "access_count", "created_at", "updated_at",
        ]
        return dict(zip(keys, row))

    @property
    def count(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
