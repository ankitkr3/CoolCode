"""Tests for cost tracking and budget management."""

import json
import threading

from coolcode.cost import BudgetStatus, CostTracker


class TestCostTracker:

    def test_calculate_cost_known_model(self):
        # MiniMax-M2.5: $0.15 in / $1.1 out per 1M tokens
        cost = CostTracker.calculate_cost("MiniMax-M2.5", 1_000_000, 1_000_000)
        assert abs(cost - 1.25) < 0.01  # 0.15 + 1.1

    def test_calculate_cost_unknown_model_uses_fallback(self):
        cost = CostTracker.calculate_cost("unknown-model", 1_000_000, 1_000_000)
        assert cost > 0  # should use fallback, not crash

    def test_record_and_session_total(self, tmp_path):
        ct = CostTracker(persist_dir=str(tmp_path))
        cost1 = ct.record("minimax", "MiniMax-M2.5", 10000, 5000)
        cost2 = ct.record("claude", "claude-sonnet-4-6", 10000, 5000)
        assert cost1 > 0
        assert cost2 > cost1  # claude is more expensive
        assert abs(ct.session_total - (cost1 + cost2)) < 0.0001

    def test_session_tokens(self, tmp_path):
        ct = CostTracker(persist_dir=str(tmp_path))
        ct.record("minimax", "MiniMax-M2.5", 1000, 500)
        ct.record("minimax", "MiniMax-M2.5", 2000, 300)
        inp, out = ct.session_tokens
        assert inp == 3000
        assert out == 800

    def test_provider_breakdown(self, tmp_path):
        ct = CostTracker(persist_dir=str(tmp_path))
        ct.record("minimax", "MiniMax-M2.5", 1000, 500)
        ct.record("claude", "claude-sonnet-4-6", 2000, 1000)
        ct.record("minimax", "MiniMax-M2.5", 500, 200)

        breakdown = ct.provider_breakdown()
        assert "minimax:MiniMax-M2.5" in breakdown
        assert "claude:claude-sonnet-4-6" in breakdown
        assert breakdown["minimax:MiniMax-M2.5"]["calls"] == 2
        assert breakdown["claude:claude-sonnet-4-6"]["calls"] == 1

    def test_persistence(self, tmp_path):
        ct = CostTracker(persist_dir=str(tmp_path))
        ct.record("minimax", "MiniMax-M2.5", 10000, 5000)

        log_path = tmp_path / "costs.jsonl"
        assert log_path.exists()
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["provider"] == "minimax"
        assert entry["input_tokens"] == 10000

    def test_thread_safety(self, tmp_path):
        ct = CostTracker(persist_dir=str(tmp_path))
        errors = []

        def record_many(thread_id):
            for i in range(50):
                try:
                    ct.record("minimax", "MiniMax-M2.5", 100, 50, task_snippet=f"t{thread_id}")
                except Exception as e:
                    errors.append(str(e))

        threads = [threading.Thread(target=record_many, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert len(ct.records) == 200  # 4 threads × 50


class TestBudgetStatus:

    def test_budget_ok_unlimited(self, tmp_path):
        ct = CostTracker(persist_dir=str(tmp_path))
        ct.record("minimax", "MiniMax-M2.5", 10000, 5000)
        status = ct.check_budget(daily_limit=0, session_limit=0)
        assert status.ok
        assert not status.exceeded

    def test_budget_exceeded_session(self, tmp_path):
        ct = CostTracker(persist_dir=str(tmp_path))
        # Record a big usage
        ct.record("claude", "claude-opus-4-6", 1_000_000, 500_000)
        status = ct.check_budget(session_limit=1.0)
        assert status.exceeded
        assert status.scope == "session"

    def test_budget_warning_at_80pct(self, tmp_path):
        ct = CostTracker(persist_dir=str(tmp_path))
        # Claude Sonnet: $3 in / $15 out per 1M
        # 100K in + 50K out = $0.3 + $0.75 = $1.05
        ct.record("claude", "claude-sonnet-4-6", 100_000, 50_000)
        status = ct.check_budget(session_limit=1.20)
        assert status.status == "warning"

    def test_budget_str_representation(self):
        bs = BudgetStatus("exceeded", "daily", 5.50, 5.00)
        s = str(bs)
        assert "exceeded" in s
        assert "daily" in s
        assert "$5.50" in s or "5.5" in s
