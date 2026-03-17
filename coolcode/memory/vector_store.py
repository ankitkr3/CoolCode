"""HNSW-based vector memory for fast semantic retrieval."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class VectorStore:
    """In-process HNSW vector store using hnswlib.

    Provides sub-millisecond approximate nearest-neighbor search
    for storing and retrieving code snippets, conversation history,
    and agent memories by semantic similarity.
    """

    def __init__(
        self,
        dim: int = 1536,
        max_elements: int = 100_000,
        persist_dir: str | None = None,
        ef_construction: int = 200,
        M: int = 16,
    ):
        import hnswlib

        self.dim = dim
        self.max_elements = max_elements
        self.persist_dir = Path(persist_dir) if persist_dir else None
        self._index = hnswlib.Index(space="cosine", dim=dim)
        self._metadata: dict[int, dict[str, Any]] = {}
        self._texts: dict[int, str] = {}
        self._next_id = 0

        index_path = self._index_path
        if index_path and index_path.exists():
            self._load()
        else:
            self._index.init_index(
                max_elements=max_elements, ef_construction=ef_construction, M=M
            )
            self._index.set_ef(50)

    @property
    def _index_path(self) -> Path | None:
        return self.persist_dir / "hnsw_index.bin" if self.persist_dir else None

    @property
    def _meta_path(self) -> Path | None:
        return self.persist_dir / "hnsw_meta.json" if self.persist_dir else None

    def _load(self) -> None:
        """Load index and metadata from disk."""
        self._index.load_index(str(self._index_path), max_elements=self.max_elements)
        self._index.set_ef(50)
        if self._meta_path and self._meta_path.exists():
            data = json.loads(self._meta_path.read_text())
            self._metadata = {int(k): v["meta"] for k, v in data.items()}
            self._texts = {int(k): v["text"] for k, v in data.items()}
            self._next_id = max(self._metadata.keys(), default=-1) + 1

    def save(self) -> None:
        """Persist index and metadata to disk."""
        if not self.persist_dir:
            return
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._index.save_index(str(self._index_path))
        data = {
            str(k): {"text": self._texts.get(k, ""), "meta": self._metadata.get(k, {})}
            for k in self._metadata
        }
        self._meta_path.write_text(json.dumps(data))

    def add(self, text: str, embedding: list[float], metadata: dict[str, Any] | None = None) -> int:
        """Add a single item to the store. Returns its ID."""
        vec = np.array([embedding], dtype=np.float32)
        item_id = self._next_id
        self._index.add_items(vec, np.array([item_id]))
        self._texts[item_id] = text
        self._metadata[item_id] = metadata or {}
        self._next_id += 1
        return item_id

    def search(
        self, query_embedding: list[float], k: int = 5
    ) -> list[dict[str, Any]]:
        """Search for the k most similar items. Returns list of {id, text, metadata, distance}."""
        if self._next_id == 0:
            return []
        vec = np.array([query_embedding], dtype=np.float32)
        actual_k = min(k, self._next_id)
        labels, distances = self._index.knn_query(vec, k=actual_k)
        results = []
        for label, dist in zip(labels[0], distances[0]):
            results.append(
                {
                    "id": int(label),
                    "text": self._texts.get(int(label), ""),
                    "metadata": self._metadata.get(int(label), {}),
                    "distance": float(dist),
                }
            )
        return results

    @property
    def count(self) -> int:
        return self._next_id

    @staticmethod
    def text_hash(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]
