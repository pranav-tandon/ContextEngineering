"""
Hybrid RAG system combining vector search with knowledge graph traversal.
"""

import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging

from ..core.base import Agent, Context, ContextComponent
from .vector.vector_database import VectorDatabase
from .graph.knowledge_graph import KnowledgeGraph
from .hybrid.keyword_index import KeywordIndex


logger = logging.getLogger(__name__)


class HybridRAG(Agent, ContextComponent):
    """
    Hybrid RAG Agent that combines vector search, knowledge graph traversal,
    and keyword matching for comprehensive retrieval.
    """
    
    def __init__(self, vector_enabled: bool = True, graph_enabled: bool = True, 
                 config: Optional[Dict[str, Any]] = None):
        Agent.__init__(self, 
            name="HybridRAG",
            description="Manages knowledge retrieval from vectors and graphs"
        )
        
        self.config = config or {}
        self.vector_enabled = vector_enabled
        self.graph_enabled = graph_enabled
        
        # Initialize components
        if self.vector_enabled:
            self.vector_store = VectorDatabase(
                self.config.get('vector_config', {})
            )
            
        if self.graph_enabled:
            self.knowledge_graph = KnowledgeGraph(
                self.config.get('graph_config', {})
            )
            
        self.keyword_index = KeywordIndex(
            self.config.get('keyword_config', {})
        )
        
        # Capabilities
        self.add_capability("retrieval")
        self.add_capability("semantic_search")
        self.add_capability("graph_traversal")
        self.add_capability("hybrid_search")
        
        # Performance tracking
        self.retrieval_stats = {
            'total_queries': 0,
            'avg_retrieval_time': 0,
            'sources_used': {'vector': 0, 'graph': 0, 'keyword': 0}
        }
        
    async def initialize(self, config: Optional[Dict[str, Any]] = None):
        """Initialize all RAG components"""
        if config:
            self.config.update(config)
            
        logger.info("Initializing Hybrid RAG system...")
        
        init_tasks = []
        
        if self.vector_enabled:
            init_tasks.append(self.vector_store.initialize())
            
        if self.graph_enabled:
            init_tasks.append(self.knowledge_graph.initialize())
            
        init_tasks.append(self.keyword_index.initialize())
        
        await asyncio.gather(*init_tasks)
        
        logger.info("Hybrid RAG system initialized")
        
    async def process(self, task: str, context: Context) -> Dict[str, Any]:
        """Process a retrieval task"""
        return await self.retrieve(task, use_graph=True)
        
    async def retrieve(self, query: str, use_graph: bool = True, 
                      max_results: int = 10, max_hops: int = 2) -> Dict[str, Any]:
        """
        Retrieve relevant information using hybrid approach.
        
        Args:
            query: The search query
            use_graph: Whether to include graph traversal
            max_results: Maximum number of results to return
            max_hops: Maximum depth for graph traversal
            
        Returns:
            Dictionary containing retrieved knowledge from all sources
        """
        start_time = datetime.now()
        
        # Prepare retrieval tasks
        retrieval_tasks = []
        
        if self.vector_enabled:
            retrieval_tasks.append(
                self._retrieve_from_vectors(query, max_results)
            )
            
        if self.graph_enabled and use_graph:
            retrieval_tasks.append(
                self._retrieve_from_graph(query, max_hops)
            )
            
        retrieval_tasks.append(
            self._retrieve_from_keywords(query, max_results)
        )
        
        # Execute retrieval in parallel
        results = await asyncio.gather(*retrieval_tasks, return_exceptions=True)
        
        # Process results
        vector_results = results[0] if self.vector_enabled and not isinstance(results[0], Exception) else []
        graph_results = results[1] if self.graph_enabled and use_graph and not isinstance(results[1], Exception) else []
        keyword_results = results[-1] if not isinstance(results[-1], Exception) else []
        
        # Merge and rank results
        merged_results = await self._merge_and_rank(
            vector_results, graph_results, keyword_results
        )
        
        # Update statistics
        retrieval_time = (datetime.now() - start_time).total_seconds()
        self._update_stats(retrieval_time, vector_results, graph_results, keyword_results)
        
        return {
            'query': query,
            'vector_results': vector_results[:max_results//3] if vector_results else [],
            'graph_results': graph_results[:max_results//3] if graph_results else [],
            'keyword_results': keyword_results[:max_results//3] if keyword_results else [],
            'merged_results': merged_results[:max_results],
            'sources': self._get_sources_used(vector_results, graph_results, keyword_results),
            'retrieval_time_ms': retrieval_time * 1000
        }
        
    async def _retrieve_from_vectors(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Retrieve from vector store using semantic search"""
        try:
            # Get query embedding
            embedding = await self.vector_store.embed_text(query)
            
            # Search vector store
            results = await self.vector_store.search(
                query_embedding=embedding,
                top_k=max_results
            )
            
            # Enhance results with metadata
            enhanced_results = []
            for result in results:
                enhanced_results.append({
                    'text': result.get('text', ''),
                    'score': result.get('score', 0),
                    'metadata': result.get('metadata', {}),
                    'source': 'vector_search',
                    'type': 'semantic'
                })
                
            return enhanced_results
            
        except Exception as e:
            logger.error(f"Error in vector retrieval: {e}")
            return []
            
    async def _retrieve_from_graph(self, query: str, max_hops: int) -> List[Dict[str, Any]]:
        """Retrieve from knowledge graph using relationship traversal"""
        try:
            # Extract entities from query
            entities = await self.knowledge_graph.extract_entities(query)
            
            if not entities:
                return []
                
            # Traverse graph from each entity
            all_paths = []
            for entity in entities[:3]:  # Limit starting entities
                paths = await self.knowledge_graph.traverse(
                    start_entity=entity,
                    query_context=query,
                    max_depth=max_hops
                )
                all_paths.extend(paths)
                
            # Convert paths to context
            graph_results = []
            for path in all_paths:
                context_text = self._path_to_text(path)
                graph_results.append({
                    'text': context_text,
                    'path': path,
                    'score': self._calculate_path_relevance(path, query),
                    'source': 'knowledge_graph',
                    'type': 'relational'
                })
                
            # Sort by relevance score
            graph_results.sort(key=lambda x: x['score'], reverse=True)
            
            return graph_results
            
        except Exception as e:
            logger.error(f"Error in graph retrieval: {e}")
            return []
            
    async def _retrieve_from_keywords(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Retrieve using keyword/exact matching"""
        try:
            results = await self.keyword_index.search(query, max_results)
            
            enhanced_results = []
            for result in results:
                enhanced_results.append({
                    'text': result.get('text', ''),
                    'score': result.get('score', 0),
                    'matches': result.get('matches', []),
                    'source': 'keyword_search',
                    'type': 'exact'
                })
                
            return enhanced_results
            
        except Exception as e:
            logger.error(f"Error in keyword retrieval: {e}")
            return []
            
    async def _merge_and_rank(self, vector_results: List[Dict[str, Any]], 
                             graph_results: List[Dict[str, Any]],
                             keyword_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Merge and rank results from different sources.
        Uses a weighted approach considering relevance and diversity.
        """
        all_results = []
        seen_content = set()
        
        # Weight configuration
        weights = {
            'vector_search': 0.4,
            'knowledge_graph': 0.4,
            'keyword_search': 0.2
        }
        
        # Process each result set
        for results, source_weight in [
            (vector_results, weights['vector_search']),
            (graph_results, weights['knowledge_graph']),
            (keyword_results, weights['keyword_search'])
        ]:
            for result in results:
                # Simple deduplication (in practice would be more sophisticated)
                content_hash = hash(result['text'][:100])
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    
                    # Adjust score based on source weight
                    weighted_score = result['score'] * source_weight
                    
                    all_results.append({
                        **result,
                        'weighted_score': weighted_score,
                        'original_score': result['score']
                    })
                    
        # Sort by weighted score
        all_results.sort(key=lambda x: x['weighted_score'], reverse=True)
        
        # Add diversity bonus to avoid too many similar results
        final_results = []
        topics_covered = set()
        
        for result in all_results:
            # Extract topic (simplified)
            topic = self._extract_topic(result['text'])
            
            # Bonus for covering new topics
            diversity_bonus = 0.1 if topic not in topics_covered else 0
            topics_covered.add(topic)
            
            result['final_score'] = result['weighted_score'] + diversity_bonus
            final_results.append(result)
            
        # Re-sort with diversity bonus
        final_results.sort(key=lambda x: x['final_score'], reverse=True)
        
        return final_results
        
    def _path_to_text(self, path: Dict[str, Any]) -> str:
        """Convert a graph path to readable text"""
        if not path:
            return ""
            
        # Build narrative from path
        entities = path.get('entities', [])
        relationships = path.get('relationships', [])
        
        if len(entities) == 1:
            return f"{entities[0]['name']}: {entities[0].get('description', '')}"
            
        text_parts = []
        for i in range(len(relationships)):
            if i < len(entities) - 1:
                text_parts.append(
                    f"{entities[i]['name']} {relationships[i]['type']} {entities[i+1]['name']}"
                )
                
        return ". ".join(text_parts)
        
    def _calculate_path_relevance(self, path: Dict[str, Any], query: str) -> float:
        """Calculate relevance score for a graph path"""
        score = 0.5  # Base score
        
        # Check entity relevance
        entities = path.get('entities', [])
        query_lower = query.lower()
        
        for entity in entities:
            if entity['name'].lower() in query_lower:
                score += 0.2
                
        # Check relationship relevance
        relationships = path.get('relationships', [])
        for rel in relationships:
            if rel['type'].lower() in query_lower:
                score += 0.1
                
        # Path length penalty (shorter paths are often more relevant)
        path_length = len(entities)
        if path_length <= 2:
            score += 0.1
        elif path_length > 4:
            score -= 0.1
            
        return min(1.0, max(0.0, score))
        
    def _extract_topic(self, text: str) -> str:
        """Extract main topic from text (simplified)"""
        # In practice, would use NLP for topic extraction
        words = text.split()[:5]  # First 5 words as simple topic
        return " ".join(words).lower()
        
    def _get_sources_used(self, vector_results: List, graph_results: List, 
                         keyword_results: List) -> List[str]:
        """Get list of sources that returned results"""
        sources = []
        if vector_results:
            sources.append("vector_store")
        if graph_results:
            sources.append("knowledge_graph")
        if keyword_results:
            sources.append("keyword_index")
        return sources
        
    def _update_stats(self, retrieval_time: float, vector_results: List,
                     graph_results: List, keyword_results: List):
        """Update retrieval statistics"""
        self.retrieval_stats['total_queries'] += 1
        
        # Update average retrieval time
        total_time = (self.retrieval_stats['avg_retrieval_time'] * 
                     (self.retrieval_stats['total_queries'] - 1))
        self.retrieval_stats['avg_retrieval_time'] = (
            (total_time + retrieval_time) / self.retrieval_stats['total_queries']
        )
        
        # Update source usage
        if vector_results:
            self.retrieval_stats['sources_used']['vector'] += 1
        if graph_results:
            self.retrieval_stats['sources_used']['graph'] += 1
        if keyword_results:
            self.retrieval_stats['sources_used']['keyword'] += 1
            
    async def add_knowledge(self, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Add new knowledge to all stores"""
        tasks = []
        
        if self.vector_enabled:
            tasks.append(self.vector_store.add_document(content, metadata))
            
        if self.graph_enabled:
            tasks.append(self.knowledge_graph.add_knowledge(content, metadata))
            
        tasks.append(self.keyword_index.add_document(content, metadata))
        
        await asyncio.gather(*tasks)
        
    async def prepare_context(self, query: str, **kwargs) -> Dict[str, Any]:
        """Prepare context for the given query (ContextComponent interface)"""
        return await self.retrieve(query, **kwargs)
        
    async def collaborate(self, other_agents: List[Agent], task: str) -> Any:
        """Collaborate with other agents"""
        collaborations = {}
        
        for agent in other_agents:
            if 'memory' in agent.capabilities:
                # Get relevant memories to enhance retrieval
                memories = await agent.process(
                    f"Recall information about: {task}",
                    Context(prompt=task)
                )
                collaborations['memory_context'] = memories
                
        return collaborations
        
    async def cleanup(self):
        """Cleanup resources"""
        cleanup_tasks = []
        
        if self.vector_enabled:
            cleanup_tasks.append(self.vector_store.cleanup())
            
        if self.graph_enabled:
            cleanup_tasks.append(self.knowledge_graph.cleanup())
            
        cleanup_tasks.append(self.keyword_index.cleanup())
        
        await asyncio.gather(*cleanup_tasks)
        
        logger.info("Hybrid RAG system cleaned up")