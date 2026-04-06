"""Token counting + budget utilities.

Uses tiktoken when available; falls back to a word-count heuristic otherwise.
Context window limits are per-family — Claude ~200K, MiniMax ~200K+, GPT ~128K.
"""

from __future__ import annotations

import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

# Approximate context windows (tokens) per model family.
# Conservative values — real limits are higher but we leave headroom for safety.
CONTEXT_WINDOWS: dict[str, int] = {
    "claude-opus-4-6": 200_000,
    "claude-sonnet-4-6": 200_000,
    "claude-haiku-4-5-20251001": 200_000,
    "MiniMax-M2.7": 245_760,
    "MiniMax-M2.7-highspeed": 245_760,
    "MiniMax-M2.5": 245_760,
    "MiniMax-M2.5-highspeed": 245_760,
}

DEFAULT_CONTEXT = 128_000


@lru_cache(maxsize=4)
def _encoder(name: str = "cl100k_base"):
    """Lazy-load tiktoken encoder. Falls back to None on failure."""
    try:
        import tiktoken

        return tiktoken.get_encoding(name)
    except Exception as e:
        logger.debug(f"tiktoken unavailable ({e}), using word-count heuristic")
        return None


def count_tokens(text: str, model: str = "") -> int:
    """Count tokens in text. Uses tiktoken cl100k_base as a universal-enough approximation."""
    if not text:
        return 0
    enc = _encoder("cl100k_base")
    if enc is not None:
        try:
            return len(enc.encode(text))
        except Exception:
            pass
    # Heuristic fallback: 1 token ≈ 0.75 words ≈ 4 chars
    return max(1, len(text) // 4)


def context_window(model: str) -> int:
    """Return the context window size (tokens) for a model, with a sane default."""
    # Try exact match, then prefix match
    if model in CONTEXT_WINDOWS:
        return CONTEXT_WINDOWS[model]
    for key, window in CONTEXT_WINDOWS.items():
        if model.startswith(key) or key.startswith(model):
            return window
    return DEFAULT_CONTEXT


def check_budget(
    prompt: str,
    model: str,
    max_output_tokens: int = 16_384,
    safety_margin: int = 2_000,
) -> tuple[bool, int, int, str]:
    """Check whether a prompt fits in the model's context window.

    Returns:
        (fits, input_tokens, available_output_tokens, warning_or_empty)
    """
    window = context_window(model)
    input_tokens = count_tokens(prompt, model)
    available = window - input_tokens - safety_margin
    if available <= 0:
        return False, input_tokens, 0, (
            f"Prompt is {input_tokens} tokens but model {model} only has a "
            f"{window}-token context. No room for output."
        )
    if available < max_output_tokens:
        return True, input_tokens, available, (
            f"Tight fit: {input_tokens} input tokens leaves only {available} for "
            f"output (wanted {max_output_tokens})."
        )
    return True, input_tokens, available, ""


def truncate_to_budget(text: str, max_tokens: int, model: str = "") -> str:
    """Truncate text to at most max_tokens, preserving the end (most relevant for context)."""
    if count_tokens(text, model) <= max_tokens:
        return text

    enc = _encoder("cl100k_base")
    if enc is not None:
        try:
            tokens = enc.encode(text)
            if len(tokens) <= max_tokens:
                return text
            # Keep the last max_tokens
            truncated = enc.decode(tokens[-max_tokens:])
            return f"[...{len(tokens) - max_tokens} tokens truncated...]\n{truncated}"
        except Exception:
            pass

    # Fallback: char-based
    char_budget = max_tokens * 4
    if len(text) <= char_budget:
        return text
    return f"[...{len(text) - char_budget} chars truncated...]\n{text[-char_budget:]}"
