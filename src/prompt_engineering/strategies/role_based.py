"""
Role-based prompting strategy - Assuming specific personas.
"""

from typing import Dict, Any, Optional
from ...core.base import PromptStrategy


class RoleBasedStrategy(PromptStrategy):
    """
    Role-based prompting: Have the model assume a specific role or persona.
    Best for creative tasks or domain-specific expertise.
    """
    
    def apply(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Apply role-based strategy.
        
        Args:
            prompt: The base prompt
            context: Optional context with role information
            
        Returns:
            Enhanced prompt with role definition
        """
        role = self._determine_role(prompt, context)
        
        # Build role-based prompt
        enhanced_parts = [
            f"You are {role}.",
            "",
            "Your expertise includes:"
        ]
        
        # Add expertise based on role
        expertise = self._get_role_expertise(role)
        for skill in expertise:
            enhanced_parts.append(f"- {skill}")
            
        enhanced_parts.extend([
            "",
            "With your expertise, please address the following:",
            "",
            prompt,
            "",
            "Provide your response drawing from your professional knowledge and experience."
        ])
        
        return "\n".join(enhanced_parts)
    
    def _determine_role(self, prompt: str, context: Optional[Dict[str, Any]]) -> str:
        """Determine appropriate role based on prompt and context"""
        # Check context for explicit role
        if context and 'role' in context:
            return context['role']
            
        # Analyze prompt to infer role
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ['code', 'program', 'software', 'debug']):
            return "an expert software engineer"
        elif any(word in prompt_lower for word in ['analyze', 'data', 'statistics', 'metrics']):
            return "a senior data analyst"
        elif any(word in prompt_lower for word in ['write', 'article', 'content', 'copy']):
            return "a professional writer"
        elif any(word in prompt_lower for word in ['design', 'ui', 'ux', 'interface']):
            return "a senior UX/UI designer"
        elif any(word in prompt_lower for word in ['research', 'study', 'investigate']):
            return "a research scientist"
        else:
            return "an expert consultant"
    
    def _get_role_expertise(self, role: str) -> List[str]:
        """Get expertise areas for a given role"""
        expertise_map = {
            "an expert software engineer": [
                "Writing clean, efficient code",
                "System design and architecture",
                "Debugging and optimization",
                "Best practices and design patterns"
            ],
            "a senior data analyst": [
                "Statistical analysis and interpretation",
                "Data visualization and reporting",
                "Pattern recognition and insights",
                "Business intelligence"
            ],
            "a professional writer": [
                "Clear and engaging communication",
                "Adapting tone for different audiences",
                "Structuring content effectively",
                "Grammar and style expertise"
            ],
            "a senior UX/UI designer": [
                "User-centered design principles",
                "Visual hierarchy and aesthetics",
                "Interaction design patterns",
                "Accessibility best practices"
            ],
            "a research scientist": [
                "Scientific methodology",
                "Critical analysis and evaluation",
                "Literature review and synthesis",
                "Hypothesis formation and testing"
            ],
            "an expert consultant": [
                "Problem-solving and analysis",
                "Strategic thinking",
                "Clear communication",
                "Domain expertise"
            ]
        }
        
        return expertise_map.get(role, expertise_map["an expert consultant"])
    
    def get_name(self) -> str:
        """Get strategy name"""
        return "role_based"