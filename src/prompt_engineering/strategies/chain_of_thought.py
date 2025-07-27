"""
Chain-of-Thought prompting strategy - Step-by-step reasoning.
"""

from typing import Dict, Any, Optional
from ...core.base import PromptStrategy


class ChainOfThoughtStrategy(PromptStrategy):
    """
    Chain-of-Thought prompting: Guide the model to think step-by-step.
    Best for complex reasoning tasks.
    """
    
    def apply(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Apply chain-of-thought strategy.
        
        Args:
            prompt: The base prompt
            context: Optional context
            
        Returns:
            Enhanced prompt encouraging step-by-step reasoning
        """
        # Check for custom template
        template = self._get_template(context)
        
        if template:
            return template.format(task=prompt)
        
        # Default chain-of-thought enhancement
        enhanced_parts = [
            prompt,
            "",
            "Let's approach this step-by-step:",
            "",
            "Step 1: Understand the problem",
            "[Analyze what is being asked]",
            "",
            "Step 2: Break down the components",  
            "[Identify key elements]",
            "",
            "Step 3: Apply reasoning",
            "[Work through the logic]",
            "",
            "Step 4: Formulate the answer",
            "[Provide the complete response]",
            "",
            "Please work through each step carefully before providing your final answer."
        ]
        
        return "\n".join(enhanced_parts)
    
    def _get_template(self, context: Optional[Dict[str, Any]]) -> Optional[str]:
        """Get custom template if available"""
        if not context or 'templates' not in context:
            return None
            
        templates = context['templates']
        return templates.get('chain_of_thought')
    
    def get_name(self) -> str:
        """Get strategy name"""
        return "chain_of_thought"