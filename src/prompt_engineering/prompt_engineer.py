"""
Prompt Engineering Agent - Intelligent prompt optimization and strategy selection.
"""

import asyncio
from typing import Dict, List, Any, Optional, Union
import json
from datetime import datetime
import logging

from ..core.base import Agent, Context, PromptStrategy
from .strategies import (
    ZeroShotStrategy,
    FewShotStrategy,
    ChainOfThoughtStrategy,
    RoleBasedStrategy,
    StructuredStrategy,
    ReActStrategy
)
from .templates.template_library import TemplateLibrary
from .optimizer.dynamic_optimizer import DynamicOptimizer


logger = logging.getLogger(__name__)


class PromptEngineer(Agent):
    """
    The Prompt Engineering Agent that crafts optimal instructions
    and learns from interactions to improve over time.
    """
    
    def __init__(self, strategies: Optional[List[str]] = None):
        super().__init__(
            name="PromptEngineer",
            description="Optimizes prompts and selects best strategies"
        )
        
        # Initialize available strategies
        self.strategies = {
            'zero_shot': ZeroShotStrategy(),
            'few_shot': FewShotStrategy(),
            'chain_of_thought': ChainOfThoughtStrategy(),
            'role_based': RoleBasedStrategy(),
            'structured': StructuredStrategy(),
            'react': ReActStrategy()
        }
        
        # Only use specified strategies if provided
        if strategies and strategies != ['default']:
            self.strategies = {
                k: v for k, v in self.strategies.items() 
                if k in strategies
            }
        
        # Initialize components
        self.template_library = TemplateLibrary()
        self.optimizer = DynamicOptimizer()
        
        # Learning data
        self.strategy_performance = {
            strategy: {
                'successes': 0,
                'failures': 0,
                'avg_quality': 0.5
            } for strategy in self.strategies
        }
        
        # Capabilities
        self.add_capability("prompt_optimization")
        self.add_capability("strategy_selection")
        self.add_capability("template_management")
        self.add_capability("performance_learning")
        
    async def initialize(self):
        """Initialize the prompt engineering agent"""
        logger.info("Initializing Prompt Engineering Agent...")
        
        # Load templates
        await self.template_library.load_templates()
        
        # Initialize optimizer
        await self.optimizer.initialize()
        
        logger.info("Prompt Engineering Agent initialized")
        
    async def process(self, task: str, context: Context) -> str:
        """Process a task and return optimized prompt"""
        # Select strategy
        strategy_name = await self.select_strategy(task)
        
        # Apply strategy
        optimized_prompt = await self.apply_strategy(task, strategy_name)
        
        # Integrate context if provided
        if context:
            optimized_prompt = await self.integrate_context(optimized_prompt, context)
            
        return optimized_prompt
    
    async def select_strategy(self, task: str) -> str:
        """
        Select the optimal prompting strategy for a given task.
        Uses both heuristics and learned performance data.
        """
        # Analyze task characteristics
        task_features = self._analyze_task(task)
        
        # Score each strategy
        strategy_scores = {}
        
        for name, strategy in self.strategies.items():
            # Base score from strategy suitability
            base_score = self._calculate_strategy_fit(strategy, task_features)
            
            # Adjust based on historical performance
            perf_data = self.strategy_performance[name]
            if perf_data['successes'] + perf_data['failures'] > 0:
                success_rate = perf_data['successes'] / (
                    perf_data['successes'] + perf_data['failures']
                )
                performance_score = success_rate * perf_data['avg_quality']
            else:
                performance_score = 0.5  # Neutral score for untested strategies
                
            # Combined score
            strategy_scores[name] = base_score * 0.7 + performance_score * 0.3
            
        # Select best strategy
        best_strategy = max(strategy_scores, key=strategy_scores.get)
        
        logger.debug(f"Selected strategy '{best_strategy}' for task with scores: {strategy_scores}")
        
        return best_strategy
    
    async def apply_strategy(self, prompt: str, strategy_name: str) -> str:
        """Apply a specific strategy to enhance a prompt"""
        if strategy_name not in self.strategies:
            logger.warning(f"Unknown strategy '{strategy_name}', using zero_shot")
            strategy_name = 'zero_shot'
            
        strategy = self.strategies[strategy_name]
        
        # Get relevant templates
        templates = await self.template_library.get_templates_for_strategy(strategy_name)
        
        # Apply strategy with templates
        enhanced_prompt = strategy.apply(prompt, {'templates': templates})
        
        # Optimize the enhanced prompt
        optimized_prompt = await self.optimizer.optimize(
            enhanced_prompt,
            strategy_name=strategy_name
        )
        
        return optimized_prompt
    
    async def integrate_context(self, prompt: str, context: Context) -> str:
        """
        Integrate full context into the prompt.
        This is where all context components come together.
        """
        sections = []
        
        # Add base prompt
        sections.append(f"# Task\n{prompt}")
        
        # Add memories if available
        if context.memories:
            memory_text = self._format_memories(context.memories)
            if memory_text:
                sections.append(f"# Relevant History\n{memory_text}")
        
        # Add knowledge if available
        if context.knowledge:
            knowledge_text = self._format_knowledge(context.knowledge)
            if knowledge_text:
                sections.append(f"# Available Knowledge\n{knowledge_text}")
        
        # Add tools if available
        if context.tools:
            tools_text = self._format_tools(context.tools)
            sections.append(f"# Available Tools\n{tools_text}")
        
        # Add agents if available
        if context.agents:
            agents_text = self._format_agents(context.agents)
            sections.append(f"# Available Agents\n{agents_text}")
        
        # Combine all sections
        integrated_prompt = "\n\n".join(sections)
        
        # Apply final optimization
        return await self.optimizer.optimize_integrated(integrated_prompt)
    
    async def collaborate(self, other_agents: List[Agent], task: str) -> Any:
        """
        Collaborate with other agents to enhance prompts.
        For example, work with RAG agent to include relevant examples.
        """
        collaborations = {}
        
        for agent in other_agents:
            if 'retrieval' in agent.capabilities:
                # Get relevant examples for few-shot prompting
                examples = await agent.process(
                    f"Find examples similar to: {task}",
                    Context(prompt=task)
                )
                collaborations['examples'] = examples
                
            elif 'memory' in agent.capabilities:
                # Get relevant past interactions
                memories = await agent.process(
                    f"Recall similar tasks: {task}",
                    Context(prompt=task)
                )
                collaborations['memories'] = memories
                
        return collaborations
    
    async def learn_from_result(self, strategy_name: str, result: Dict[str, Any]):
        """Learn from the result of using a specific strategy"""
        if strategy_name not in self.strategy_performance:
            return
            
        perf_data = self.strategy_performance[strategy_name]
        
        # Update success/failure counts
        if result.get('success', False):
            perf_data['successes'] += 1
        else:
            perf_data['failures'] += 1
            
        # Update quality score (simplified - in practice would be more sophisticated)
        quality_score = self._calculate_quality_score(result)
        
        # Running average of quality
        total_attempts = perf_data['successes'] + perf_data['failures']
        perf_data['avg_quality'] = (
            (perf_data['avg_quality'] * (total_attempts - 1) + quality_score) 
            / total_attempts
        )
        
        # Let optimizer learn from the interaction
        await self.optimizer.learn(strategy_name, result)
        
    def _analyze_task(self, task: str) -> Dict[str, Any]:
        """Analyze task characteristics"""
        return {
            'length': len(task),
            'complexity': task.count(' ') / 10,  # Simple word count heuristic
            'has_examples': 'example' in task.lower() or 'e.g.' in task,
            'is_creative': any(word in task.lower() for word in ['create', 'write', 'design', 'imagine']),
            'is_analytical': any(word in task.lower() for word in ['analyze', 'evaluate', 'compare', 'assess']),
            'needs_reasoning': any(word in task.lower() for word in ['why', 'how', 'explain', 'reason']),
            'needs_structure': any(word in task.lower() for word in ['list', 'steps', 'format', 'structure']),
            'needs_tools': any(word in task.lower() for word in ['calculate', 'search', 'fetch', 'api'])
        }
    
    def _calculate_strategy_fit(self, strategy: PromptStrategy, features: Dict[str, Any]) -> float:
        """Calculate how well a strategy fits the task features"""
        score = 0.5  # Base score
        
        strategy_name = strategy.get_name()
        
        if strategy_name == 'zero_shot' and features['length'] < 100:
            score += 0.2
        elif strategy_name == 'few_shot' and features['has_examples']:
            score += 0.3
        elif strategy_name == 'chain_of_thought' and features['needs_reasoning']:
            score += 0.4
        elif strategy_name == 'role_based' and features['is_creative']:
            score += 0.3
        elif strategy_name == 'structured' and features['needs_structure']:
            score += 0.3
        elif strategy_name == 'react' and features['needs_tools']:
            score += 0.4
            
        # Penalize complexity mismatch
        if features['complexity'] > 5 and strategy_name == 'zero_shot':
            score -= 0.2
            
        return max(0, min(1, score))
    
    def _calculate_quality_score(self, result: Dict[str, Any]) -> float:
        """Calculate quality score from result"""
        score = 0.5
        
        # Factors that indicate quality
        if result.get('token_count', 0) < 1000:  # Efficient
            score += 0.1
        if result.get('latency_ms', float('inf')) < 1000:  # Fast
            score += 0.1
        if result.get('user_satisfaction', 0) > 0:  # If available
            score = score * 0.5 + result['user_satisfaction'] * 0.5
            
        return min(1.0, score)
    
    def _format_memories(self, memories: Dict[str, Any]) -> str:
        """Format memories for inclusion in prompt"""
        if not memories or not memories.get('relevant'):
            return ""
            
        memory_lines = []
        for memory in memories['relevant'][:3]:  # Limit to top 3
            memory_lines.append(f"- {memory.get('summary', str(memory))}")
            
        return "\n".join(memory_lines)
    
    def _format_knowledge(self, knowledge: Dict[str, Any]) -> str:
        """Format knowledge for inclusion in prompt"""
        if not knowledge:
            return ""
            
        knowledge_lines = []
        
        # Format vector search results
        if 'vector_results' in knowledge:
            knowledge_lines.append("Retrieved Information:")
            for result in knowledge['vector_results'][:3]:
                knowledge_lines.append(f"- {result.get('text', '')[:200]}...")
                
        # Format graph relationships
        if 'graph_results' in knowledge:
            knowledge_lines.append("\nRelated Concepts:")
            for relation in knowledge['graph_results'][:3]:
                knowledge_lines.append(
                    f"- {relation.get('entity1')} → {relation.get('relationship')} → {relation.get('entity2')}"
                )
                
        return "\n".join(knowledge_lines)
    
    def _format_tools(self, tools: List[str]) -> str:
        """Format available tools for inclusion in prompt"""
        if not tools:
            return "No tools available"
            
        return "\n".join([f"- {tool}" for tool in tools])
    
    def _format_agents(self, agents: List[str]) -> str:
        """Format available agents for inclusion in prompt"""
        if not agents:
            return "No specialist agents available"
            
        return "\n".join([f"- {agent}" for agent in agents])
    
    async def cleanup(self):
        """Cleanup resources"""
        await self.optimizer.save_learning_data()
        logger.info("Prompt Engineering Agent cleaned up")