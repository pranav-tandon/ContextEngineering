"""
Zero-shot prompting strategy - Direct instruction without examples.
"""

from typing import Dict, Any, Optional
from ...core.base import PromptStrategy


class ZeroShotStrategy(PromptStrategy):
    """
    Zero-shot prompting: Clear, direct instructions without examples.
    Best for simple, straightforward tasks.
    """
    
    def apply(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Apply zero-shot strategy to enhance the prompt.
        
        Args:
            prompt: The base prompt
            context: Optional context with templates
            
        Returns:
            Enhanced prompt
        """
        # Get template if available
        template = None
        if context and 'templates' in context:
            templates = context['templates']
            if templates and 'zero_shot' in templates:
                template = templates['zero_shot']
        
        if template:
            # Use template
            enhanced = template.format(task=prompt)
        else:
            # Default zero-shot enhancement
            enhanced = self._default_enhancement(prompt)
            
        return enhanced
    
    def _default_enhancement(self, prompt: str) -> str:
        """Apply default zero-shot enhancements"""
        # Add clarity and specificity
        enhanced_parts = []
        
        # Add instruction clarity
        enhanced_parts.append("Please complete the following task:")
        enhanced_parts.append("")
        enhanced_parts.append(prompt)
        enhanced_parts.append("")
        enhanced_parts.append("Provide a clear and concise response.")
        
        return "\n".join(enhanced_parts)
    
    def get_name(self) -> str:
        """Get strategy name"""
        return "zero_shot"