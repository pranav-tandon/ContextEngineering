"""
Few-shot prompting strategy - Learning from examples.
"""

from typing import Dict, Any, Optional, List
from ...core.base import PromptStrategy


class FewShotStrategy(PromptStrategy):
    """
    Few-shot prompting: Provide examples to guide the model.
    Best for tasks where patterns can be learned from examples.
    """
    
    def apply(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Apply few-shot strategy with examples.
        
        Args:
            prompt: The base prompt
            context: Optional context with examples and templates
            
        Returns:
            Enhanced prompt with examples
        """
        examples = self._get_examples(context)
        
        # Build few-shot prompt
        enhanced_parts = []
        
        # Add instruction
        enhanced_parts.append("Learn from these examples and then complete the task:")
        enhanced_parts.append("")
        
        # Add examples
        if examples:
            enhanced_parts.append("Examples:")
            for i, example in enumerate(examples, 1):
                enhanced_parts.append(f"Example {i}:")
                enhanced_parts.append(f"Input: {example.get('input', '')}")
                enhanced_parts.append(f"Output: {example.get('output', '')}")
                enhanced_parts.append("")
        
        # Add the actual task
        enhanced_parts.append("Now complete this task:")
        enhanced_parts.append(f"Input: {prompt}")
        enhanced_parts.append("Output:")
        
        return "\n".join(enhanced_parts)
    
    def _get_examples(self, context: Optional[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Extract or generate examples from context"""
        if not context:
            return self._default_examples()
            
        # Check for provided examples
        if 'examples' in context:
            return context['examples'][:3]  # Limit to 3 examples
            
        # Check templates for examples
        if 'templates' in context and 'few_shot_examples' in context['templates']:
            return context['templates']['few_shot_examples']
            
        return self._default_examples()
    
    def _default_examples(self) -> List[Dict[str, str]]:
        """Provide generic default examples"""
        return [
            {
                "input": "Summarize: The cat sat on the mat. It was a sunny day.",
                "output": "A cat rested on a mat during sunny weather."
            },
            {
                "input": "Summarize: The stock market rose by 2% today. Investors are optimistic.",
                "output": "Markets gained 2% amid investor optimism."
            }
        ]
    
    def get_name(self) -> str:
        """Get strategy name"""
        return "few_shot"