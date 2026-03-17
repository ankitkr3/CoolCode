"""Configuration management for Cool Code."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LLMConfig:
    """Configuration for a single LLM provider."""

    provider: str  # "claude" or "minimax"
    model: str
    api_key: str = ""
    group_id: str = ""  # MiniMax only
    temperature: float = 0.0
    max_tokens: int = 16384
    timeout: int = 120


@dataclass
class SwarmConfig:
    """Configuration for the swarm system."""

    num_workers: int = 3
    consensus_algorithm: str = "majority"  # majority, raft, byzantine, gossip, weighted
    queen_type: str = "coordinator"  # coordinator, evaluator, delegator
    parallel_racing: bool = True
    fault_tolerance_ratio: float = 0.33  # f < n/3 for byzantine


@dataclass
class MemoryConfig:
    """Configuration for the memory system."""

    sqlite_path: str = ""
    vector_dim: int = 1536
    vector_max_elements: int = 100_000
    lru_cache_size: int = 1024
    scopes: list[str] = field(default_factory=lambda: ["project", "local", "user"])


@dataclass
class CoolCodeConfig:
    """Root configuration for Cool Code."""

    project_dir: str = "."
    providers: list[LLMConfig] = field(default_factory=list)
    swarm: SwarmConfig = field(default_factory=SwarmConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    max_retries: int = 3
    verbose: bool = False

    @classmethod
    def from_env(cls, project_dir: str = ".") -> CoolCodeConfig:
        """Build config from environment variables."""
        providers: list[LLMConfig] = []

        # Claude provider
        anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
        if anthropic_key:
            providers.append(
                LLMConfig(
                    provider="claude",
                    model=os.getenv("COOLCODE_CLAUDE_MODEL", "claude-sonnet-4-6"),
                    api_key=anthropic_key,
                )
            )

        # MiniMax provider
        minimax_key = os.getenv("MINIMAX_API_KEY", "")
        if minimax_key:
            providers.append(
                LLMConfig(
                    provider="minimax",
                    model=os.getenv("COOLCODE_MINIMAX_MODEL", "MiniMax-M2.5"),
                    api_key=minimax_key,
                    group_id=os.getenv("MINIMAX_GROUP_ID", ""),
                )
            )

        project_path = Path(project_dir).resolve()
        sqlite_path = str(project_path / ".coolcode" / "memory.db")

        return cls(
            project_dir=str(project_path),
            providers=providers,
            memory=MemoryConfig(sqlite_path=sqlite_path),
        )

    def get_provider(self, name: str | None = None) -> LLMConfig:
        """Get a specific provider config, or the default."""
        if not self.providers:
            raise ValueError("No LLM providers configured. Set ANTHROPIC_API_KEY or MINIMAX_API_KEY.")
        if name is None:
            default = os.getenv("COOLCODE_DEFAULT_PROVIDER", "claude")
            for p in self.providers:
                if p.provider == default:
                    return p
            return self.providers[0]
        for p in self.providers:
            if p.provider == name:
                return p
        raise ValueError(f"Provider '{name}' not configured.")
