"""
Structured prompting strategy - Using templates and schemas.
"""

from typing import Dict, Any, Optional
from ...core.base import PromptStrategy


class StructuredStrategy(PromptStrategy):
    """
    Structured prompting: Use specific formats and schemas.
    Best for tasks requiring specific output formats.
    """
    
    def apply(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Apply structured strategy with format specifications.
        
        Args:
            prompt: The base prompt
            context: Optional context with structure requirements
            
        Returns:
            Enhanced prompt with structure guidelines
        """
        # Determine required structure
        structure_type = self._determine_structure(prompt, context)
        
        # Build structured prompt
        enhanced_parts = [
            prompt,
            "",
            f"Please provide your response in the following {structure_type} format:",
            ""
        ]
        
        # Add structure template
        template = self._get_structure_template(structure_type)
        enhanced_parts.append(template)
        
        enhanced_parts.extend([
            "",
            "Ensure your response follows this exact structure."
        ])
        
        return "\n".join(enhanced_parts)
    
    def _determine_structure(self, prompt: str, context: Optional[Dict[str, Any]]) -> str:
        """Determine what type of structure is needed"""
        # Check context for explicit structure
        if context and 'structure' in context:
            return context['structure']
            
        # Analyze prompt to infer structure
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ['list', 'enumerate', 'items']):
            return "list"
        elif any(word in prompt_lower for word in ['compare', 'contrast', 'versus']):
            return "comparison"
        elif any(word in prompt_lower for word in ['steps', 'process', 'procedure']):
            return "step-by-step"
        elif any(word in prompt_lower for word in ['pros', 'cons', 'advantages']):
            return "pros-and-cons"
        elif any(word in prompt_lower for word in ['analyze', 'report', 'assessment']):
            return "analytical-report"
        else:
            return "structured-response"
    
    def _get_structure_template(self, structure_type: str) -> str:
        """Get template for specific structure type"""
        templates = {
            "list": """1. [First item]
   - Details about this item
   
2. [Second item]
   - Details about this item
   
3. [Third item]
   - Details about this item
   
(Continue as needed...)""",
            
            "comparison": """## Option A: [Name]
**Strengths:**
- [Strength 1]
- [Strength 2]

**Weaknesses:**
- [Weakness 1]
- [Weakness 2]

## Option B: [Name]
**Strengths:**
- [Strength 1]
- [Strength 2]

**Weaknesses:**
- [Weakness 1]
- [Weakness 2]

## Recommendation:
[Your recommendation with reasoning]""",
            
            "step-by-step": """### Step 1: [Action]
- Description: [What to do]
- Expected outcome: [Result]

### Step 2: [Action]
- Description: [What to do]
- Expected outcome: [Result]

### Step 3: [Action]
- Description: [What to do]
- Expected outcome: [Result]

(Continue with additional steps...)""",
            
            "pros-and-cons": """## Pros:
✓ [Advantage 1]
  - Explanation: [Why this is beneficial]
  
✓ [Advantage 2]
  - Explanation: [Why this is beneficial]
  
✓ [Advantage 3]
  - Explanation: [Why this is beneficial]

## Cons:
✗ [Disadvantage 1]
  - Explanation: [Why this is a drawback]
  
✗ [Disadvantage 2]
  - Explanation: [Why this is a drawback]
  
✗ [Disadvantage 3]
  - Explanation: [Why this is a drawback]

## Summary:
[Overall assessment]""",
            
            "analytical-report": """# Analysis Report

## Executive Summary
[Brief overview of findings]

## Key Findings
1. [Finding 1]
2. [Finding 2]
3. [Finding 3]

## Detailed Analysis
### Section 1: [Topic]
[Detailed analysis]

### Section 2: [Topic]
[Detailed analysis]

## Recommendations
1. [Recommendation 1]
2. [Recommendation 2]

## Conclusion
[Summary and next steps]""",
            
            "structured-response": """## Overview
[Brief introduction to your response]

## Main Points
### Point 1: [Title]
[Detailed explanation]

### Point 2: [Title]
[Detailed explanation]

### Point 3: [Title]
[Detailed explanation]

## Conclusion
[Summary of key takeaways]"""
        }
        
        return templates.get(structure_type, templates["structured-response"])
    
    def get_name(self) -> str:
        """Get strategy name"""
        return "structured"