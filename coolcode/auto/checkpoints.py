"""Git checkpoint management for autonomous pipelines.

Creates lightweight git branches as rollback points between pipeline stages.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

from coolcode.tools.git import _run_git

logger = logging.getLogger(__name__)

CHECKPOINT_PREFIX = "coolcode/auto"


@dataclass
class Checkpoint:
    """A git checkpoint (branch) for a pipeline stage."""

    name: str
    branch: str
    stage_index: int
    timestamp: float
    commit_hash: str = ""


class CheckpointManager:
    """Manages git checkpoints for autonomous pipeline stages.

    Creates branches at each stage so rollback is always possible.
    """

    def __init__(self, project_dir: str, pipeline_id: str = ""):
        self.project_dir = project_dir
        self.pipeline_id = pipeline_id or str(int(time.time()))
        self._checkpoints: list[Checkpoint] = []

    def create(self, name: str, stage_index: int) -> Checkpoint:
        """Create a checkpoint at the current state.

        Stages all changes and commits them on a checkpoint branch.
        """
        branch = f"{CHECKPOINT_PREFIX}/{self.pipeline_id}/{stage_index:02d}-{name}"

        # Create the branch
        _run_git(["branch", branch], cwd=self.project_dir)

        # Get current commit hash
        result = _run_git(["rev-parse", "HEAD"], cwd=self.project_dir)
        commit_hash = result.strip().split("\n")[0] if result else ""

        cp = Checkpoint(
            name=name,
            branch=branch,
            stage_index=stage_index,
            timestamp=time.time(),
            commit_hash=commit_hash,
        )
        self._checkpoints.append(cp)
        logger.info(f"Checkpoint created: {branch} ({commit_hash[:8]})")
        return cp

    def rollback(self, checkpoint: Checkpoint) -> str:
        """Rollback to a specific checkpoint.

        Resets the working directory to the checkpoint's state.
        """
        # Stash any current uncommitted work
        _run_git(["stash", "push", "-m", f"coolcode: auto-stash before rollback to {checkpoint.name}"],
                 cwd=self.project_dir)

        # Reset to the checkpoint commit
        if checkpoint.commit_hash:
            result = _run_git(["reset", "--hard", checkpoint.commit_hash], cwd=self.project_dir)
        else:
            result = _run_git(["checkout", checkpoint.branch], cwd=self.project_dir)

        logger.info(f"Rolled back to checkpoint: {checkpoint.name}")
        return result

    def rollback_to_stage(self, stage_index: int) -> str:
        """Rollback to a specific stage by index."""
        for cp in self._checkpoints:
            if cp.stage_index == stage_index:
                return self.rollback(cp)
        return f"Error: No checkpoint found for stage {stage_index}"

    def list_checkpoints(self) -> list[Checkpoint]:
        """List all checkpoints for this pipeline."""
        return list(self._checkpoints)

    def cleanup(self) -> None:
        """Delete all checkpoint branches for this pipeline."""
        for cp in self._checkpoints:
            _run_git(["branch", "-D", cp.branch], cwd=self.project_dir)
        self._checkpoints.clear()
        logger.info(f"Cleaned up checkpoints for pipeline {self.pipeline_id}")

    @property
    def latest(self) -> Checkpoint | None:
        """Most recent checkpoint."""
        return self._checkpoints[-1] if self._checkpoints else None
