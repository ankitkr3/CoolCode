"""Configuration management for Cool Code — with persistent config file support."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

CONFIG_DIR = Path.home() / ".coolcode"
CONFIG_FILE = CONFIG_DIR / "config.json"


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

    def to_dict(self) -> dict:
        return {
            "provider": self.provider,
            "model": self.model,
            "api_key": self.api_key,
            "group_id": self.group_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> LLMConfig:
        return cls(
            provider=data["provider"],
            model=data["model"],
            api_key=data.get("api_key", ""),
            group_id=data.get("group_id", ""),
        )


# Available models per provider
AVAILABLE_MODELS: dict[str, list[dict[str, str]]] = {
    "claude": [
        {"id": "claude-sonnet-4-6", "name": "Claude Sonnet 4.6", "tier": "Balanced"},
        {"id": "claude-opus-4-6", "name": "Claude Opus 4.6", "tier": "Highest quality"},
        {"id": "claude-haiku-4-5-20251001", "name": "Claude Haiku 4.5", "tier": "Fastest"},
    ],
    "minimax": [
        {"id": "MiniMax-M2.7-highspeed", "name": "MiniMax M2.7 Highspeed", "tier": "Fastest"},
        {"id": "MiniMax-M2.7", "name": "MiniMax M2.7", "tier": "Best quality"},
        {"id": "MiniMax-M2.5", "name": "MiniMax M2.5", "tier": "Best value"},
        {"id": "MiniMax-M2.5-highspeed", "name": "MiniMax M2.5 Highspeed", "tier": "Fast & cheap"},
    ],
}


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
    vector_dim: int = 384  # MiniLM embedding dimension
    vector_max_elements: int = 100_000
    lru_cache_size: int = 1024
    scopes: list[str] = field(default_factory=lambda: ["project", "local", "user"])


@dataclass
class CostConfig:
    """Configuration for cost tracking and budget caps."""

    budget_daily_usd: float = 0.0  # 0 = unlimited
    budget_session_usd: float = 0.0  # 0 = unlimited


@dataclass
class CoolCodeConfig:
    """Root configuration for Cool Code."""

    project_dir: str = "."
    providers: list[LLMConfig] = field(default_factory=list)
    swarm: SwarmConfig = field(default_factory=SwarmConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    max_retries: int = 3
    verbose: bool = False

    def save(self) -> None:
        """Persist provider config to ~/.coolcode/config.json."""
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        data = {
            "providers": [p.to_dict() for p in self.providers],
        }
        CONFIG_FILE.write_text(json.dumps(data, indent=2))

    @classmethod
    def load_saved(cls) -> list[LLMConfig]:
        """Load providers from saved config file."""
        if not CONFIG_FILE.exists():
            return []
        try:
            data = json.loads(CONFIG_FILE.read_text())
            return [LLMConfig.from_dict(p) for p in data.get("providers", [])]
        except (json.JSONDecodeError, KeyError):
            return []

    @classmethod
    def from_env(cls, project_dir: str = ".") -> CoolCodeConfig:
        """Build config from saved config file, then override with env vars if set."""
        # 1. Load from persistent config file first
        saved_providers = cls.load_saved()

        # 2. Override with env vars if they are set (env vars take priority)
        providers: list[LLMConfig] = []

        anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
        minimax_key = os.getenv("MINIMAX_API_KEY", "")

        if anthropic_key:
            # Find saved claude config for model preference, or use default
            saved_claude = next((p for p in saved_providers if p.provider == "claude"), None)
            providers.append(
                LLMConfig(
                    provider="claude",
                    model=os.getenv("COOLCODE_CLAUDE_MODEL", saved_claude.model if saved_claude else "claude-sonnet-4-6"),
                    api_key=anthropic_key,
                )
            )

        if minimax_key:
            saved_minimax = next((p for p in saved_providers if p.provider == "minimax"), None)
            providers.append(
                LLMConfig(
                    provider="minimax",
                    model=os.getenv("COOLCODE_MINIMAX_MODEL", saved_minimax.model if saved_minimax else "MiniMax-M2.7"),
                    api_key=minimax_key,
                    group_id=os.getenv("MINIMAX_GROUP_ID", saved_minimax.group_id if saved_minimax else ""),
                )
            )

        # 3. If no env vars, use saved config as-is
        if not providers and saved_providers:
            providers = saved_providers

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
            raise ValueError("No LLM providers configured. Run `coolcode` to set up.")
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
