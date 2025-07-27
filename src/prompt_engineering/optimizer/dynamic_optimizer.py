"""
Dynamic Prompt Optimizer - Learns and optimizes prompts based on performance.
"""

import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import asyncio
from collections import defaultdict
import re


class DynamicOptimizer:
    """
    Optimizes prompts dynamically based on learned patterns and performance data.
    """
    
    def __init__(self):
        # Performance tracking
        self.optimization_history = defaultdict(list)
        self.pattern_effectiveness = defaultdict(float)
        
        # Optimization rules
        self.optimization_rules = self._get_default_rules()
        
        # Token optimization settings
        self.max_tokens = 8000
        self.compression_enabled = True
        
    async def initialize(self):
        """Initialize the optimizer"""
        # Load historical optimization data if available
        await self._load_optimization_history()
        
    async def optimize(self, prompt: str, strategy_name: Optional[str] = None) -> str:
        """
        Optimize a prompt for clarity, effectiveness, and token efficiency.
        """
        optimized = prompt
        
        # Apply rule-based optimizations
        for rule in self.optimization_rules:
            if rule['enabled'] and self._should_apply_rule(rule, strategy_name):
                optimized = self._apply_rule(optimized, rule)
                
        # Apply learned optimizations
        optimized = await self._apply_learned_patterns(optimized, strategy_name)
        
        # Compress if needed
        if self.compression_enabled:
            optimized = self._compress_prompt(optimized)
            
        # Track optimization
        self._track_optimization(prompt, optimized, strategy_name)
        
        return optimized
    
    async def optimize_integrated(self, integrated_prompt: str) -> str:
        """
        Optimize an integrated prompt with full context.
        Special optimization for combined prompts.
        """
        # Remove redundancy
        optimized = self._remove_redundancy(integrated_prompt)
        
        # Optimize section ordering
        optimized = self._optimize_section_order(optimized)
        
        # Compress while maintaining clarity
        if len(optimized) > self.max_tokens * 4:  # Rough char to token estimate
            optimized = self._intelligent_compression(optimized)
            
        return optimized
    
    async def learn(self, strategy_name: str, result: Dict[str, Any]):
        """Learn from the result of using a prompt"""
        # Extract optimization patterns that worked well
        if result.get('success') and result.get('quality_score', 0) > 0.8:
            prompt_used = result.get('prompt', '')
            patterns = self._extract_patterns(prompt_used)
            
            for pattern in patterns:
                self.pattern_effectiveness[pattern] += 0.1
                
        # Store in optimization history
        self.optimization_history[strategy_name].append({
            'timestamp': datetime.now().isoformat(),
            'success': result.get('success'),
            'quality_score': result.get('quality_score', 0),
            'token_count': result.get('token_count', 0)
        })
        
    def _get_default_rules(self) -> List[Dict[str, Any]]:
        """Get default optimization rules"""
        return [
            {
                'name': 'clarity_enhancement',
                'enabled': True,
                'patterns': [
                    (r'please\\s+', ''),  # Remove unnecessary "please"
                    (r'\\s+', ' '),  # Normalize whitespace
                    (r'\\n{3,}', '\\n\\n'),  # Limit consecutive newlines
                ],
                'strategies': ['all']
            },
            {
                'name': 'instruction_clarity',
                'enabled': True,
                'replacements': {
                    'do the following': 'complete this task',
                    'in order to': 'to',
                    'at this point in time': 'now',
                    'due to the fact that': 'because'
                },
                'strategies': ['all']
            },
            {
                'name': 'token_efficiency',
                'enabled': True,
                'strategies': ['all']
            }
        ]
        
    def _should_apply_rule(self, rule: Dict[str, Any], strategy_name: Optional[str]) -> bool:
        """Check if a rule should be applied"""
        if 'all' in rule['strategies']:
            return True
        return strategy_name in rule.get('strategies', [])
        
    def _apply_rule(self, prompt: str, rule: Dict[str, Any]) -> str:
        """Apply an optimization rule"""
        optimized = prompt
        
        if rule['name'] == 'clarity_enhancement':
            for pattern, replacement in rule.get('patterns', []):
                optimized = re.sub(pattern, replacement, optimized)
                
        elif rule['name'] == 'instruction_clarity':
            for old, new in rule.get('replacements', {}).items():
                optimized = optimized.replace(old, new)
                
        return optimized
        
    async def _apply_learned_patterns(self, prompt: str, strategy_name: Optional[str]) -> str:
        """Apply patterns learned from successful interactions"""
        # Sort patterns by effectiveness
        effective_patterns = sorted(
            self.pattern_effectiveness.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        optimized = prompt
        
        # Apply top effective patterns
        for pattern, score in effective_patterns[:5]:
            if score > 0.6:  # Only apply highly effective patterns
                # This is simplified - real implementation would be more sophisticated
                if pattern == 'bullet_points' and '\\n-' not in optimized:
                    # Convert lists to bullet points
                    optimized = re.sub(r'(\\d+\\.)', r'-', optimized)
                elif pattern == 'emphasis' and '**' not in optimized:
                    # Add emphasis to key terms
                    key_terms = self._extract_key_terms(optimized)
                    for term in key_terms[:3]:
                        optimized = optimized.replace(term, f'**{term}**', 1)
                        
        return optimized
        
    def _compress_prompt(self, prompt: str) -> str:
        """Compress prompt to save tokens while maintaining meaning"""
        compressed = prompt
        
        # Remove excessive examples
        example_pattern = r'Example \\d+:.*?(?=Example \\d+:|$)'
        examples = re.findall(example_pattern, compressed, re.DOTALL)
        if len(examples) > 3:
            # Keep only first 3 examples
            compressed = re.sub(
                r'(Example [123]:.*?)(Example [4-9]\\d*:.*)',
                r'\\1',
                compressed,
                flags=re.DOTALL
            )
            
        # Simplify verbose instructions
        verbose_patterns = [
            (r'In order to accomplish this task, you should', 'To do this,'),
            (r'Please make sure to', ''),
            (r'It is important that you', 'You must'),
            (r'You are requested to', ''),
        ]
        
        for pattern, replacement in verbose_patterns:
            compressed = re.sub(pattern, replacement, compressed, flags=re.IGNORECASE)
            
        return compressed.strip()
        
    def _remove_redundancy(self, prompt: str) -> str:
        """Remove redundant information from integrated prompt"""
        lines = prompt.split('\\n')
        seen_content = set()
        unique_lines = []
        
        for line in lines:
            # Simple content hash (in practice would be more sophisticated)
            content_hash = line.strip().lower()
            if content_hash and content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_lines.append(line)
                
        return '\\n'.join(unique_lines)
        
    def _optimize_section_order(self, prompt: str) -> str:
        """Optimize the order of sections in integrated prompt"""
        # Parse sections
        sections = self._parse_sections(prompt)
        
        # Optimal order: Task -> Context -> Examples -> Instructions
        optimal_order = ['task', 'context', 'examples', 'knowledge', 'tools', 'instructions']
        
        # Reorder sections
        ordered_sections = []
        for section_type in optimal_order:
            if section_type in sections:
                ordered_sections.append(sections[section_type])
                
        # Add any remaining sections
        for section_type, content in sections.items():
            if section_type not in optimal_order:
                ordered_sections.append(content)
                
        return '\\n\\n'.join(ordered_sections)
        
    def _intelligent_compression(self, prompt: str) -> str:
        """Intelligently compress long prompts"""
        # This is a simplified version
        # Real implementation would use more sophisticated NLP
        
        compressed = prompt
        
        # Summarize long knowledge sections
        if '# Available Knowledge' in compressed:
            knowledge_start = compressed.find('# Available Knowledge')
            knowledge_end = compressed.find('\\n#', knowledge_start + 1)
            if knowledge_end == -1:
                knowledge_end = len(compressed)
                
            knowledge_section = compressed[knowledge_start:knowledge_end]
            if len(knowledge_section) > 2000:
                # Keep only most relevant parts
                compressed = (
                    compressed[:knowledge_start] +
                    '# Available Knowledge\\n[Relevant context provided]\\n' +
                    compressed[knowledge_end:]
                )
                
        return compressed
        
    def _extract_patterns(self, prompt: str) -> List[str]:
        """Extract successful patterns from a prompt"""
        patterns = []
        
        if '\\n-' in prompt or '\\n*' in prompt:
            patterns.append('bullet_points')
        if '**' in prompt:
            patterns.append('emphasis')
        if 'Step 1:' in prompt or '1.' in prompt:
            patterns.append('numbered_steps')
        if '# ' in prompt:
            patterns.append('headers')
            
        return patterns
        
    def _extract_key_terms(self, prompt: str) -> List[str]:
        """Extract key terms from prompt"""
        # Simple extraction - real implementation would use NLP
        words = prompt.split()
        
        # Filter for likely important terms
        key_terms = []
        for word in words:
            if (len(word) > 5 and 
                word[0].isupper() and 
                word not in ['Please', 'Provide', 'Create', 'Generate']):
                key_terms.append(word)
                
        return key_terms[:5]
        
    def _parse_sections(self, prompt: str) -> Dict[str, str]:
        """Parse prompt into sections"""
        sections = {}
        current_section = 'task'
        current_content = []
        
        for line in prompt.split('\\n'):
            if line.startswith('# '):
                # Save previous section
                if current_content:
                    sections[current_section] = '\\n'.join(current_content)
                    
                # Start new section
                section_name = line[2:].lower()
                if 'task' in section_name:
                    current_section = 'task'
                elif 'knowledge' in section_name:
                    current_section = 'knowledge'
                elif 'tool' in section_name:
                    current_section = 'tools'
                elif 'example' in section_name:
                    current_section = 'examples'
                else:
                    current_section = section_name
                    
                current_content = [line]
            else:
                current_content.append(line)
                
        # Save final section
        if current_content:
            sections[current_section] = '\\n'.join(current_content)
            
        return sections
        
    def _track_optimization(self, original: str, optimized: str, strategy_name: Optional[str]):
        """Track optimization for analysis"""
        reduction = len(original) - len(optimized)
        if reduction > 0:
            compression_ratio = reduction / len(original)
            # Could log or store this data for analysis
            
    async def _load_optimization_history(self):
        """Load historical optimization data"""
        # In a real implementation, this would load from persistent storage
        pass
        
    async def save_learning_data(self):
        """Save learned optimization patterns"""
        # In a real implementation, this would persist to storage
        learning_data = {
            'pattern_effectiveness': dict(self.pattern_effectiveness),
            'optimization_history': dict(self.optimization_history),
            'timestamp': datetime.now().isoformat()
        }
        # Would save to file or database