"""
Context Engineering: A Unified Framework for AI Context Management

This framework unifies Prompt Engineering, RAG (with Knowledge Graphs), 
MCP, A2A protocols, and Memory systems into a single platform.
"""

__version__ = "0.1.0"
__author__ = "Context Engineering Team"

from .core.engine import UnifiedContextEngine
from .prompt_engineering.prompt_engineer import PromptEngineer
from .rag.hybrid_rag import HybridRAG
from .memory.hierarchical_memory import HierarchicalMemory

__all__ = [
    "UnifiedContextEngine",
    "PromptEngineer", 
    "HybridRAG",
    "HierarchicalMemory"
]