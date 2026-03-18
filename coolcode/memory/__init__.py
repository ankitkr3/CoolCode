"""Memory subsystem — vector store, knowledge graph, collective memory, scoped memory, embeddings."""

from coolcode.memory.collective import CollectiveMemory
from coolcode.memory.embeddings import EMBEDDING_DIM, embed_text, embed_batch
from coolcode.memory.knowledge_graph import KnowledgeGraph
from coolcode.memory.learning_bridge import LearningBridge
from coolcode.memory.scoped import ScopedMemory
from coolcode.memory.vector_store import VectorStore

__all__ = [
    "VectorStore",
    "KnowledgeGraph",
    "CollectiveMemory",
    "ScopedMemory",
    "LearningBridge",
    "EMBEDDING_DIM",
    "embed_text",
    "embed_batch",
]
