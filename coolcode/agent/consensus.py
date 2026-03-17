"""Consensus algorithms for multi-agent decision making.

Implements 5 algorithms:
- Majority: simple >50% vote
- Weighted: votes weighted by agent confidence/track record
- Raft: leader-based consensus with term elections
- Byzantine: BFT tolerant of f < n/3 faulty agents
- Gossip: epidemic-style convergence for eventual consistency
"""

from __future__ import annotations

import logging
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Vote:
    """A single agent's vote on a proposal."""

    agent_id: str
    choice: str  # The chosen option or answer
    confidence: float = 1.0  # 0.0 - 1.0
    reasoning: str = ""


@dataclass
class ConsensusResult:
    """The outcome of a consensus round."""

    algorithm: str
    winner: str | None
    votes: list[Vote]
    agreement_ratio: float
    converged: bool
    rounds: int = 1
    details: dict[str, Any] = field(default_factory=dict)


class ConsensusEngine:
    """Multi-algorithm consensus engine for swarm decision-making."""

    def __init__(self, algorithm: str = "majority", fault_tolerance: float = 0.33):
        self.algorithm = algorithm
        self.fault_tolerance = fault_tolerance
        self._dispatch = {
            "majority": self._majority,
            "weighted": self._weighted,
            "raft": self._raft,
            "byzantine": self._byzantine,
            "gossip": self._gossip,
        }

    def resolve(self, votes: list[Vote]) -> ConsensusResult:
        """Run the configured consensus algorithm on a set of votes."""
        if not votes:
            return ConsensusResult(
                algorithm=self.algorithm,
                winner=None,
                votes=[],
                agreement_ratio=0.0,
                converged=False,
            )
        fn = self._dispatch.get(self.algorithm)
        if fn is None:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
        return fn(votes)

    def _majority(self, votes: list[Vote]) -> ConsensusResult:
        """Simple majority vote — winner needs >50%."""
        counts = Counter(v.choice for v in votes)
        total = len(votes)
        winner, top_count = counts.most_common(1)[0]
        ratio = top_count / total
        return ConsensusResult(
            algorithm="majority",
            winner=winner if ratio > 0.5 else None,
            votes=votes,
            agreement_ratio=ratio,
            converged=ratio > 0.5,
        )

    def _weighted(self, votes: list[Vote]) -> ConsensusResult:
        """Weighted vote — each vote scaled by agent confidence."""
        weighted_counts: dict[str, float] = {}
        total_weight = 0.0
        for v in votes:
            weighted_counts[v.choice] = weighted_counts.get(v.choice, 0.0) + v.confidence
            total_weight += v.confidence
        winner = max(weighted_counts, key=weighted_counts.get)
        ratio = weighted_counts[winner] / total_weight if total_weight > 0 else 0.0
        return ConsensusResult(
            algorithm="weighted",
            winner=winner,
            votes=votes,
            agreement_ratio=ratio,
            converged=ratio > 0.5,
            details={"weighted_counts": weighted_counts},
        )

    def _raft(self, votes: list[Vote]) -> ConsensusResult:
        """Raft-style: highest-confidence agent becomes leader, others follow.
        Leader's choice wins if majority of followers don't strongly disagree.
        """
        leader = max(votes, key=lambda v: v.confidence)
        followers = [v for v in votes if v.agent_id != leader.agent_id]
        agreeing = [v for v in followers if v.choice == leader.choice]
        # Leader + agreeing followers vs total
        support = 1 + len(agreeing)
        total = len(votes)
        ratio = support / total
        return ConsensusResult(
            algorithm="raft",
            winner=leader.choice if ratio > 0.5 else None,
            votes=votes,
            agreement_ratio=ratio,
            converged=ratio > 0.5,
            details={"leader": leader.agent_id, "term": 1},
        )

    def _byzantine(self, votes: list[Vote]) -> ConsensusResult:
        """Byzantine fault-tolerant: tolerates f < n/3 faulty agents.
        Requires 2f+1 agreement for consensus.
        """
        n = len(votes)
        f = int(n * self.fault_tolerance)
        required = 2 * f + 1
        counts = Counter(v.choice for v in votes)
        winner, top_count = counts.most_common(1)[0]
        converged = top_count >= required
        return ConsensusResult(
            algorithm="byzantine",
            winner=winner if converged else None,
            votes=votes,
            agreement_ratio=top_count / n,
            converged=converged,
            details={"n": n, "f": f, "required": required, "got": top_count},
        )

    def _gossip(self, votes: list[Vote]) -> ConsensusResult:
        """Gossip/epidemic: simulate rounds of random pair-wise exchange.
        Agents update their choice toward the majority in each round.
        Converges when all agents agree or max rounds reached.
        """
        max_rounds = 10
        # Start with each agent's initial choice
        state = {v.agent_id: v.choice for v in votes}
        agents = list(state.keys())

        for round_num in range(1, max_rounds + 1):
            # Each agent gossips with a random peer
            new_state = dict(state)
            for agent in agents:
                peer = random.choice([a for a in agents if a != agent]) if len(agents) > 1 else agent
                # Adopt peer's choice with probability proportional to local majority
                local_counts = Counter(state.values())
                if local_counts[state[peer]] > local_counts[state[agent]]:
                    new_state[agent] = state[peer]
            state = new_state
            # Check convergence
            unique = set(state.values())
            if len(unique) == 1:
                winner = unique.pop()
                return ConsensusResult(
                    algorithm="gossip",
                    winner=winner,
                    votes=votes,
                    agreement_ratio=1.0,
                    converged=True,
                    rounds=round_num,
                )

        # Return best after max rounds
        final_counts = Counter(state.values())
        winner, top_count = final_counts.most_common(1)[0]
        return ConsensusResult(
            algorithm="gossip",
            winner=winner,
            votes=votes,
            agreement_ratio=top_count / len(agents),
            converged=False,
            rounds=max_rounds,
        )
