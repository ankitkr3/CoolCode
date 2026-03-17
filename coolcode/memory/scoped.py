"""3-scope agent memory: project, local, and user — with cross-agent transfer."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ScopedMemory:
    """Memory scoped at three levels:

    - **project**: Shared across all agents working on this project.
      Stored in `<project>/.coolcode/memory/project/`.
    - **local**: Specific to this machine/environment.
      Stored in `<project>/.coolcode/memory/local/`.
    - **user**: Global user preferences and learnings, shared across all projects.
      Stored in `~/.coolcode/memory/user/`.

    Each memory is a JSON file keyed by a human-readable name.
    Supports cross-agent transfer: any agent can read/write any scope.
    """

    SCOPES = ("project", "local", "user")

    def __init__(self, project_dir: str):
        self._project_dir = Path(project_dir).resolve()
        self._roots = {
            "project": self._project_dir / ".coolcode" / "memory" / "project",
            "local": self._project_dir / ".coolcode" / "memory" / "local",
            "user": Path.home() / ".coolcode" / "memory" / "user",
        }
        for root in self._roots.values():
            root.mkdir(parents=True, exist_ok=True)

    def _path(self, scope: str, key: str) -> Path:
        if scope not in self.SCOPES:
            raise ValueError(f"Invalid scope: {scope}. Must be one of {self.SCOPES}")
        return self._roots[scope] / f"{key}.json"

    def store(self, scope: str, key: str, value: Any, agent_id: str = "unknown") -> None:
        """Store a memory in the given scope."""
        path = self._path(scope, key)
        data = {
            "key": key,
            "scope": scope,
            "value": value,
            "agent_id": agent_id,
        }
        path.write_text(json.dumps(data, indent=2))

    def recall(self, scope: str, key: str) -> Any | None:
        """Retrieve a memory from the given scope. Returns None if not found."""
        path = self._path(scope, key)
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        return data.get("value")

    def recall_all(self, scope: str) -> dict[str, Any]:
        """Retrieve all memories from a scope."""
        root = self._roots[scope]
        result: dict[str, Any] = {}
        for f in root.glob("*.json"):
            try:
                data = json.loads(f.read_text())
                result[data["key"]] = data.get("value")
            except (json.JSONDecodeError, KeyError):
                logger.warning(f"Corrupt memory file: {f}")
        return result

    def delete(self, scope: str, key: str) -> bool:
        """Delete a memory. Returns True if it existed."""
        path = self._path(scope, key)
        if path.exists():
            path.unlink()
            return True
        return False

    def transfer(self, from_scope: str, to_scope: str, key: str) -> bool:
        """Transfer a memory between scopes (cross-agent transfer)."""
        value = self.recall(from_scope, key)
        if value is None:
            return False
        self.store(to_scope, key, value, agent_id="transfer")
        return True

    def list_keys(self, scope: str) -> list[str]:
        """List all memory keys in a scope."""
        root = self._roots[scope]
        return [f.stem for f in root.glob("*.json")]

    def search(self, query: str, scope: str | None = None) -> list[dict[str, Any]]:
        """Simple text search across memories."""
        scopes = [scope] if scope else list(self.SCOPES)
        results: list[dict[str, Any]] = []
        query_lower = query.lower()
        for s in scopes:
            for key in self.list_keys(s):
                value = self.recall(s, key)
                text = json.dumps(value) if not isinstance(value, str) else value
                if query_lower in text.lower() or query_lower in key.lower():
                    results.append({"scope": s, "key": key, "value": value})
        return results
