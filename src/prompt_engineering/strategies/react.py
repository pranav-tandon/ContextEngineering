"""
ReAct prompting strategy - Reasoning and Acting interleaved.
"""

from typing import Dict, Any, Optional, List
from ...core.base import PromptStrategy


class ReActStrategy(PromptStrategy):
    """
    ReAct prompting: Interleave reasoning and action steps.
    Best for tasks that require tool use or multi-step problem solving.
    """
    
    def apply(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Apply ReAct strategy for reasoning and acting.
        
        Args:
            prompt: The base prompt
            context: Optional context with available tools
            
        Returns:
            Enhanced prompt with ReAct format
        """
        # Get available tools/actions
        available_tools = self._get_available_tools(context)
        
        # Build ReAct prompt
        enhanced_parts = [
            "You will solve this task by interleaving Thought, Action, and Observation steps.",
            "",
            "Available Actions:"
        ]
        
        # List available tools
        for tool in available_tools:
            enhanced_parts.append(f"- {tool}")
            
        enhanced_parts.extend([
            "",
            "Format your response as:",
            "Thought: [Your reasoning about what to do next]",
            "Action: [The action/tool to use]",
            "Observation: [Result of the action]",
            "... (repeat as needed)",
            "Final Answer: [Your final response]",
            "",
            "Task:",
            prompt,
            "",
            "Begin solving:"
        ])
        
        return "\n".join(enhanced_parts)
    
    def _get_available_tools(self, context: Optional[Dict[str, Any]]) -> List[str]:
        """Get list of available tools/actions"""
        # Check context for tools
        if context and 'tools' in context:
            return context['tools']
            
        # Default tools for demonstration
        return [
            "Search[query] - Search for information",
            "Calculate[expression] - Perform calculations",
            "Lookup[key] - Look up specific information",
            "Think[reasoning] - Internal reasoning step"
        ]
    
    def get_name(self) -> str:
        """Get strategy name"""
        return "react"