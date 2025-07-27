"""
Unified Context Engine - The main orchestrator for context engineering.
"""

import asyncio
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import json
import logging

from .base import (
    Context, QueryResult, TaskComplexity,
    ContextComponent, Agent
)
from ..prompt_engineering.prompt_engineer import PromptEngineer
from ..rag.hybrid_rag import HybridRAG
from ..mcp.client import MCPClient
from ..a2a.coordinator import A2ACoordinator
from ..memory.hierarchical_memory import HierarchicalMemory
from ..orchestrator.context_orchestrator import ContextOrchestrator


logger = logging.getLogger(__name__)


class UnifiedContextEngine:
    """
    The main engine that unifies all context engineering components.
    Orchestrates prompt engineering, RAG, MCP, A2A, and memory systems.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Unified Context Engine.
        
        Args:
            config: Configuration dictionary with component settings
        """
        self.config = config or {}
        
        # Initialize all components as agents
        self.prompt_agent = PromptEngineer(
            strategies=self.config.get('prompt_strategies', ['default'])
        )
        
        self.rag_agent = HybridRAG(
            vector_enabled=self.config.get('rag_vector', True),
            graph_enabled=self.config.get('rag_graph', True),
            config=self.config.get('rag_config', {})
        )
        
        self.mcp_agent = MCPClient(
            enabled=self.config.get('mcp_enabled', True),
            config=self.config.get('mcp_config', {})
        )
        
        self.a2a_coordinator = A2ACoordinator(
            enabled=self.config.get('a2a_enabled', True),
            config=self.config.get('a2a_config', {})
        )
        
        self.memory_agent = HierarchicalMemory(
            config=self.config.get('memory_config', {})
        )
        
        self.orchestrator = ContextOrchestrator(
            config=self.config.get('orchestrator_config', {})
        )
        
        # Performance tracking
        self.metrics = {
            'queries_processed': 0,
            'average_latency': 0,
            'success_rate': 1.0,
            'token_usage': []
        }
        
        self.initialized = False
        
    async def initialize(self):
        """Initialize all components"""
        if self.initialized:
            return
            
        logger.info("Initializing Unified Context Engine...")
        
        # Initialize all components in parallel
        await asyncio.gather(
            self.prompt_agent.initialize(),
            self.rag_agent.initialize(),
            self.mcp_agent.initialize(),
            self.a2a_coordinator.initialize(),
            self.memory_agent.initialize(),
            self.orchestrator.initialize()
        )
        
        self.initialized = True
        logger.info("Unified Context Engine initialized successfully")
        
    async def query(self, request: str, user_id: Optional[str] = None, **kwargs) -> QueryResult:
        """
        Process a query with full context engineering.
        
        Args:
            request: The user's request/query
            user_id: Optional user identifier for personalization
            **kwargs: Additional parameters
            
        Returns:
            QueryResult with response and context details
        """
        if not self.initialized:
            await self.initialize()
            
        start_time = datetime.now()
        
        try:
            # Step 1: Analyze and optimize the request with prompt engineering
            logger.debug("Step 1: Optimizing prompt")
            prompt_strategy = await self.prompt_agent.select_strategy(request)
            optimized_request = await self.prompt_agent.apply_strategy(
                request, prompt_strategy
            )
            
            # Step 2: Retrieve memories
            logger.debug("Step 2: Retrieving memories")
            memories = await self.memory_agent.recall(optimized_request, user_id)
            
            # Step 3: Analyze request complexity
            complexity = await self._analyze_complexity(optimized_request)
            logger.debug(f"Request complexity: {complexity}")
            
            # Step 4: Build context in parallel
            context_tasks = []
            
            if complexity.needs_retrieval:
                context_tasks.append(
                    self.rag_agent.retrieve(
                        optimized_request,
                        use_graph=complexity.needs_relationships
                    )
                )
            
            if complexity.needs_tools:
                context_tasks.append(
                    self.mcp_agent.prepare_tools(optimized_request)
                )
                
            if complexity.needs_agents:
                context_tasks.append(
                    self.a2a_coordinator.summon_agents(optimized_request)
                )
                
            # Execute context building in parallel
            context_results = await asyncio.gather(*context_tasks, return_exceptions=True)
            
            # Step 5: Build unified context
            context = Context(
                prompt=optimized_request,
                memories=memories,
                knowledge=context_results[0] if len(context_results) > 0 and not isinstance(context_results[0], Exception) else None,
                tools=context_results[1] if len(context_results) > 1 and not isinstance(context_results[1], Exception) else None,
                agents=context_results[2] if len(context_results) > 2 and not isinstance(context_results[2], Exception) else None,
                metadata={
                    'user_id': user_id,
                    'complexity': complexity.value,
                    'strategy': prompt_strategy
                }
            )
            
            # Step 6: Create final prompt with full context
            final_prompt = await self.prompt_agent.integrate_context(
                optimized_request, context
            )
            
            # Step 7: Execute with orchestrator
            result = await self.orchestrator.execute(final_prompt, context)
            
            # Step 8: Update memories
            await self.memory_agent.remember(
                interaction={
                    'request': request,
                    'response': result['response'],
                    'context': context,
                    'success': True
                },
                user_id=user_id,
                importance_score=self._calculate_importance(result)
            )
            
            # Step 9: Learn from the interaction
            await self.prompt_agent.learn_from_result(prompt_strategy, result)
            
            # Calculate metrics
            latency = (datetime.now() - start_time).total_seconds() * 1000
            token_count = result.get('token_count', 0)
            
            # Update performance metrics
            self._update_metrics(latency, token_count, success=True)
            
            return QueryResult(
                response=result['response'],
                context_summary=self._summarize_context(context),
                prompt_strategy=prompt_strategy,
                knowledge_sources=self._get_knowledge_sources(context),
                memory_stats=memories.get('stats', {}),
                token_count=token_count,
                latency_ms=latency,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            
            # Update metrics for failure
            latency = (datetime.now() - start_time).total_seconds() * 1000
            self._update_metrics(latency, 0, success=False)
            
            return QueryResult(
                response="I encountered an error processing your request.",
                context_summary={},
                prompt_strategy="fallback",
                knowledge_sources=[],
                memory_stats={},
                token_count=0,
                latency_ms=latency,
                success=False,
                error=str(e)
            )
    
    async def _analyze_complexity(self, request: str) -> TaskComplexity:
        """Analyze the complexity of a request"""
        # Simple heuristic-based analysis for now
        # In a real implementation, this would use ML models
        
        complexity_score = 0
        
        # Check for indicators of complexity
        if len(request) > 200:
            complexity_score += 1
        if any(word in request.lower() for word in ['analyze', 'compare', 'evaluate', 'design']):
            complexity_score += 2
        if any(word in request.lower() for word in ['implement', 'create', 'build', 'develop']):
            complexity_score += 2
        if '?' in request:
            complexity_score += 1
            
        # Determine complexity level and needs
        if complexity_score >= 4:
            complexity = TaskComplexity.EXPERT
        elif complexity_score >= 2:
            complexity = TaskComplexity.COMPLEX
        elif complexity_score >= 1:
            complexity = TaskComplexity.MODERATE
        else:
            complexity = TaskComplexity.SIMPLE
            
        # Add attributes for different needs
        complexity.needs_retrieval = complexity_score >= 1
        complexity.needs_tools = 'implement' in request.lower() or 'create' in request.lower()
        complexity.needs_agents = complexity == TaskComplexity.EXPERT
        complexity.needs_relationships = 'how' in request.lower() or 'why' in request.lower()
        complexity.value = complexity_score
        
        return complexity
    
    def _summarize_context(self, context: Context) -> Dict[str, Any]:
        """Create a summary of the context used"""
        summary = {
            'prompt_length': len(context.prompt),
            'has_memories': context.memories is not None,
            'has_knowledge': context.knowledge is not None,
            'has_tools': context.tools is not None and len(context.tools) > 0,
            'has_agents': context.agents is not None and len(context.agents) > 0
        }
        
        if context.memories:
            summary['memory_count'] = len(context.memories.get('relevant', []))
            
        if context.knowledge:
            summary['knowledge_sources'] = len(context.knowledge.get('sources', []))
            
        return summary
    
    def _get_knowledge_sources(self, context: Context) -> List[str]:
        """Extract knowledge source names from context"""
        sources = []
        
        if context.knowledge:
            sources.extend(context.knowledge.get('sources', []))
            
        if context.memories:
            sources.append('memory_system')
            
        return sources
    
    def _calculate_importance(self, result: Dict[str, Any]) -> float:
        """Calculate importance score for memory storage"""
        # Simple heuristic - in practice this would be more sophisticated
        score = 0.5
        
        if result.get('token_count', 0) > 1000:
            score += 0.2
            
        if result.get('tools_used', 0) > 0:
            score += 0.1
            
        if result.get('agents_involved', 0) > 0:
            score += 0.2
            
        return min(score, 1.0)
    
    def _update_metrics(self, latency: float, tokens: int, success: bool):
        """Update performance metrics"""
        self.metrics['queries_processed'] += 1
        
        # Update average latency
        total_latency = self.metrics['average_latency'] * (self.metrics['queries_processed'] - 1)
        self.metrics['average_latency'] = (total_latency + latency) / self.metrics['queries_processed']
        
        # Update success rate
        if success:
            current_successes = self.metrics['success_rate'] * (self.metrics['queries_processed'] - 1)
            self.metrics['success_rate'] = (current_successes + 1) / self.metrics['queries_processed']
        else:
            current_successes = self.metrics['success_rate'] * (self.metrics['queries_processed'] - 1)
            self.metrics['success_rate'] = current_successes / self.metrics['queries_processed']
            
        # Track token usage
        self.metrics['token_usage'].append(tokens)
        
    async def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            **self.metrics,
            'total_tokens': sum(self.metrics['token_usage']),
            'average_tokens': sum(self.metrics['token_usage']) / len(self.metrics['token_usage']) if self.metrics['token_usage'] else 0
        }
    
    async def shutdown(self):
        """Gracefully shutdown all components"""
        logger.info("Shutting down Unified Context Engine...")
        
        await asyncio.gather(
            self.prompt_agent.cleanup(),
            self.rag_agent.cleanup(),
            self.mcp_agent.cleanup(),
            self.a2a_coordinator.cleanup(),
            self.memory_agent.cleanup(),
            self.orchestrator.cleanup()
        )
        
        logger.info("Unified Context Engine shut down successfully")