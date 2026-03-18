"""Local embeddings using sentence-transformers — no API calls needed.

Uses all-MiniLM-L6-v2 (22MB) for fast, local vector generation.
Falls back to simple TF-IDF-style hashing if model unavailable.
"""

from __future__ import annotations

import hashlib
import logging
import math
from collections import Counter
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Lazy-loaded model
_model = None
_model_failed = False


def _get_model():
    """Lazy-load the sentence-transformers model."""
    global _model, _model_failed
    if _model is not None:
        return _model
    if _model_failed:
        return None
    try:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Loaded local embedding model: all-MiniLM-L6-v2")
        return _model
    except ImportError:
        logger.warning("sentence-transformers not installed, using hash-based embeddings")
        _model_failed = True
        return None
    except Exception as e:
        logger.warning(f"Failed to load embedding model: {e}, using hash-based embeddings")
        _model_failed = True
        return None


def embed_text(text: str, dim: int = 384) -> list[float]:
    """Generate embedding for text.

    Uses MiniLM if available (384-dim), otherwise falls back to
    deterministic hash-based embeddings.

    Args:
        text: Text to embed
        dim: Embedding dimension (384 for MiniLM, ignored if model available)

    Returns:
        List of floats representing the embedding vector
    """
    model = _get_model()
    if model is not None:
        embedding = model.encode(text, normalize_embeddings=True)
        return embedding.tolist()
    return _hash_embed(text, dim)


def embed_batch(texts: list[str], dim: int = 384) -> list[list[float]]:
    """Batch embedding — much faster than calling embed_text() in a loop."""
    model = _get_model()
    if model is not None:
        embeddings = model.encode(texts, normalize_embeddings=True, batch_size=32)
        return embeddings.tolist()
    return [_hash_embed(t, dim) for t in texts]


def _hash_embed(text: str, dim: int = 384) -> list[float]:
    """Deterministic hash-based embedding fallback.

    Not semantic, but gives consistent vectors for identical text.
    Good enough for exact-match retrieval and basic clustering.
    """
    # Tokenize
    words = text.lower().split()
    if not words:
        return [0.0] * dim

    # Create a pseudo-embedding by hashing word n-grams
    vec = np.zeros(dim, dtype=np.float32)
    for i, word in enumerate(words):
        # Hash each word to a position and value
        h = int(hashlib.md5(word.encode()).hexdigest(), 16)
        pos = h % dim
        val = ((h >> 8) % 1000) / 1000.0 - 0.5  # value between -0.5 and 0.5
        vec[pos] += val

        # Also hash bigrams for better discrimination
        if i > 0:
            bigram = f"{words[i-1]}_{word}"
            h2 = int(hashlib.md5(bigram.encode()).hexdigest(), 16)
            pos2 = h2 % dim
            vec[pos2] += ((h2 >> 8) % 1000) / 1000.0 - 0.5

    # Normalize to unit length
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm

    return vec.tolist()


# Embedding dimension — 384 for MiniLM, used by VectorStore
EMBEDDING_DIM = 384
