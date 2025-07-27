"""
Base classes and interfaces for the Context Engineering framework.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import asyncio
from datetime import datetime


class TaskComplexity(Enum):
    """Complexity levels for tasks"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXPERT = "expert"


@dataclass
class Context:
    """Represents the complete context for an AI interaction"""
    prompt: str
    memories: Optional[Dict[str, Any]] = None
    knowledge: Optional[Dict[str, Any]] = None
    tools: Optional[List[str]] = None
    agents: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class ContextDesign:
    """Represents a designed context architecture"""
    architecture: Dict[str, Any]
    implementation: str
    estimated_performance: Dict[str, float]
    explanation: str


@dataclass
class QueryResult:
    """Result from a context-engineered query"""
    response: str
    context_summary: Dict[str, Any]
    prompt_strategy: str
    knowledge_sources: List[str]
    memory_stats: Dict[str, int]
    token_count: int
    latency_ms: float
    success: bool = True
    error: Optional[str] = None


class Agent(ABC):
    """Base class for all agents in the ecosystem"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.capabilities = []
        
    @abstractmethod
    async def process(self, task: str, context: Context) -> Any:
        """Process a task with given context"""
        pass
    
    @abstractmethod
    async def collaborate(self, other_agents: List['Agent'], task: str) -> Any:
        """Collaborate with other agents"""
        pass
    
    def add_capability(self, capability: str):
        """Add a capability to this agent"""
        self.capabilities.append(capability)


class ContextComponent(ABC):
    """Base class for context components (RAG, MCP, etc.)"""
    
    @abstractmethod
    async def initialize(self, config: Dict[str, Any]):
        """Initialize the component with configuration"""
        pass
    
    @abstractmethod
    async def prepare_context(self, query: str, **kwargs) -> Dict[str, Any]:
        """Prepare context for the given query"""
        pass
    
    @abstractmethod
    async def cleanup(self):
        """Cleanup resources"""
        pass


class PromptStrategy(ABC):
    """Base class for prompt engineering strategies"""
    
    @abstractmethod
    def apply(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Apply the strategy to enhance the prompt"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get strategy name"""
        pass


class MemoryStore(ABC):
    """Base class for memory storage backends"""
    
    @abstractmethod
    async def store(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None):
        """Store a memory"""
        pass
    
    @abstractmethod
    async def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve a memory by key"""
        pass
    
    @abstractmethod
    async def search(self, query: str, top_k: int = 10) -> List[Any]:
        """Search memories by query"""
        pass
    
    @abstractmethod
    async def delete(self, key: str):
        """Delete a memory"""
        pass


class VectorStore(ABC):
    """Base class for vector storage backends"""
    
    @abstractmethod
    async def add_documents(self, documents: List[Dict[str, Any]], embeddings: List[List[float]]):
        """Add documents with their embeddings"""
        pass
    
    @abstractmethod
    async def search(self, query_embedding: List[float], top_k: int = 10) -> List[Dict[str, Any]]:
        """Search by embedding similarity"""
        pass
    
    @abstractmethod
    async def delete(self, doc_ids: List[str]):
        """Delete documents by IDs"""
        pass


class GraphDatabase(ABC):
    """Base class for knowledge graph databases"""
    
    @abstractmethod
    async def add_entity(self, entity: Dict[str, Any]):
        """Add an entity to the graph"""
        pass
    
    @abstractmethod
    async def add_relationship(self, entity1_id: str, entity2_id: str, relationship: str, properties: Optional[Dict] = None):
        """Add a relationship between entities"""
        pass
    
    @abstractmethod
    async def traverse(self, start_entity: str, pattern: str, max_depth: int = 3) -> List[Dict[str, Any]]:
        """Traverse the graph from a starting entity"""
        pass
    
    @abstractmethod
    async def query(self, cypher_query: str) -> List[Dict[str, Any]]:
        """Execute a graph query"""
        pass