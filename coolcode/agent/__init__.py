"""Agent subsystem — swarm intelligence, queen coordination, worker agents, consensus."""

from coolcode.agent.consensus import ConsensusEngine
from coolcode.agent.queen import QueenAgent
from coolcode.agent.router import TaskRouter
from coolcode.agent.swarm import Swarm
from coolcode.agent.worker import WorkerAgent, WorkerType

__all__ = [
    "Swarm",
    "QueenAgent",
    "WorkerAgent",
    "WorkerType",
    "ConsensusEngine",
    "TaskRouter",
]
