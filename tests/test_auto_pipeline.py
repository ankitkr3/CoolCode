"""Tests for autonomous pipeline, gates, and checkpoints."""

import os
import subprocess
from unittest.mock import MagicMock, patch

from coolcode.auto.checkpoints import CheckpointManager
from coolcode.auto.gates import GateDecision, GateManager, GateResult
from coolcode.auto.pipeline import AutoPipeline, PipelinePlan, PipelineStage, STAGE_TEMPLATES
from coolcode.agent.worker import WorkerType


class TestPipelinePlan:

    def test_stage_templates_exist(self):
        assert len(STAGE_TEMPLATES) >= 4
        # First stage should be planning
        assert "plan" in STAGE_TEMPLATES[0].description.lower()
        # Should have a gate after planning
        assert STAGE_TEMPLATES[0].is_gate

    def test_plan_creates_stages(self):
        from coolcode.agent.swarm import Swarm
        with patch.object(Swarm, "__init__", lambda self, **kwargs: None):
            swarm = Swarm.__new__(Swarm)
            swarm.config = MagicMock()
            swarm.config.providers = [MagicMock(model="MiniMax-M2.5")]
            swarm._cancelled = False
            swarm._goal = "general"

            pipeline = AutoPipeline(swarm=swarm)
            plan = pipeline.plan("build a login page")

            assert plan.total_stages == len(STAGE_TEMPLATES)
            assert plan.task == "build a login page"
            assert plan.estimated_total_cost > 0
            # Task should be embedded in stage prompts
            assert "login page" in plan.stages[0].prompt

    def test_gate_frequency(self):
        from coolcode.agent.swarm import Swarm
        with patch.object(Swarm, "__init__", lambda self, **kwargs: None):
            swarm = Swarm.__new__(Swarm)
            swarm.config = MagicMock()
            swarm.config.providers = [MagicMock(model="MiniMax-M2.5")]
            swarm._cancelled = False

            # Gate every stage
            pipeline = AutoPipeline(swarm=swarm, gate_frequency=1)
            plan = pipeline.plan("test")
            gates = [s for s in plan.stages if s.is_gate]
            assert len(gates) == plan.total_stages

            # No gates
            pipeline_no_gates = AutoPipeline(swarm=swarm, gate_frequency=0)
            plan_ng = pipeline_no_gates.plan("test")
            # Only default gates from templates
            template_gates = sum(1 for t in STAGE_TEMPLATES if t.is_gate)
            actual_gates = sum(1 for s in plan_ng.stages if s.is_gate)
            assert actual_gates == template_gates


class TestGateDecision:

    def test_gate_decision_values(self):
        assert GateDecision.APPROVE == "approve"
        assert GateDecision.REJECT == "reject"
        assert GateDecision.EDIT == "edit"
        assert GateDecision.ROLLBACK == "rollback"

    def test_gate_result_with_edit(self):
        result = GateResult(
            decision=GateDecision.EDIT,
            edit_instructions="use OAuth instead of JWT"
        )
        assert result.decision == GateDecision.EDIT
        assert "OAuth" in result.edit_instructions


class TestCheckpointManager:

    def test_create_checkpoint(self, tmp_path):
        # Init a git repo
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmp_path, capture_output=True)
        subprocess.run(["git", "config", "user.name", "test"], cwd=tmp_path, capture_output=True)
        (tmp_path / "file.txt").write_text("hello")
        subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, capture_output=True)

        cm = CheckpointManager(str(tmp_path), pipeline_id="test-001")
        cp = cm.create("plan", stage_index=0)

        assert cp.name == "plan"
        assert cp.stage_index == 0
        assert "coolcode/auto/test-001" in cp.branch
        assert len(cm.list_checkpoints()) == 1

    def test_list_and_cleanup(self, tmp_path):
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmp_path, capture_output=True)
        subprocess.run(["git", "config", "user.name", "test"], cwd=tmp_path, capture_output=True)
        (tmp_path / "file.txt").write_text("hello")
        subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, capture_output=True)

        cm = CheckpointManager(str(tmp_path), pipeline_id="test-002")
        cm.create("stage-1", 0)
        cm.create("stage-2", 1)
        assert len(cm.list_checkpoints()) == 2

        cm.cleanup()
        assert len(cm.list_checkpoints()) == 0

    def test_latest_checkpoint(self, tmp_path):
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmp_path, capture_output=True)
        subprocess.run(["git", "config", "user.name", "test"], cwd=tmp_path, capture_output=True)
        (tmp_path / "file.txt").write_text("hello")
        subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, capture_output=True)

        cm = CheckpointManager(str(tmp_path))
        assert cm.latest is None

        cm.create("first", 0)
        cm.create("second", 1)
        assert cm.latest.name == "second"
