"""
Template Library for prompt engineering strategies.
"""

import json
import os
from typing import Dict, List, Any, Optional
import asyncio
from pathlib import Path


class TemplateLibrary:
    """
    Manages prompt templates for different strategies and use cases.
    """
    
    def __init__(self, templates_dir: Optional[str] = None):
        self.templates_dir = templates_dir or os.path.join(
            os.path.dirname(__file__), 'data'
        )
        self.templates = {}
        self.loaded = False
        
    async def load_templates(self):
        """Load templates from storage"""
        if self.loaded:
            return
            
        # Load built-in templates
        self.templates = self._get_builtin_templates()
        
        # Load custom templates if directory exists
        templates_path = Path(self.templates_dir)
        if templates_path.exists():
            for file_path in templates_path.glob("*.json"):
                try:
                    with open(file_path, 'r') as f:
                        custom_templates = json.load(f)
                        self.templates.update(custom_templates)
                except Exception as e:
                    print(f"Error loading template {file_path}: {e}")
                    
        self.loaded = True
        
    async def get_templates_for_strategy(self, strategy_name: str) -> Dict[str, Any]:
        """Get templates for a specific strategy"""
        if not self.loaded:
            await self.load_templates()
            
        return self.templates.get(strategy_name, {})
    
    async def add_template(self, strategy: str, name: str, template: str):
        """Add a new template"""
        if strategy not in self.templates:
            self.templates[strategy] = {}
            
        self.templates[strategy][name] = template
        
        # Optionally persist to disk
        await self._save_templates()
        
    async def _save_templates(self):
        """Save templates to disk"""
        os.makedirs(self.templates_dir, exist_ok=True)
        
        templates_file = os.path.join(self.templates_dir, "custom_templates.json")
        with open(templates_file, 'w') as f:
            json.dump(self.templates, f, indent=2)
            
    def _get_builtin_templates(self) -> Dict[str, Dict[str, str]]:
        """Get built-in templates for each strategy"""
        return {
            "zero_shot": {
                "default": "Task: {task}\n\nPlease provide a clear, direct response.",
                "technical": "Technical Task: {task}\n\nProvide a precise, technical response with relevant details.",
                "creative": "Creative Task: {task}\n\nBe creative and think outside the box in your response."
            },
            
            "few_shot": {
                "default": "Learn from these examples:\n\n{examples}\n\nNow complete: {task}",
                "classification": "Classification Examples:\n{examples}\n\nClassify: {task}",
                "transformation": "Transformation Examples:\n{examples}\n\nTransform: {task}"
            },
            
            "chain_of_thought": {
                "default": "{task}\n\nLet's think step by step:\n1. First, ...\n2. Then, ...\n3. Finally, ...",
                "math": "{task}\n\nLet's solve this step-by-step:\nStep 1: Identify what we know\nStep 2: Determine what we need to find\nStep 3: Apply the appropriate method\nStep 4: Calculate\nStep 5: Verify the answer",
                "analysis": "{task}\n\nLet's analyze this systematically:\n1. Context and Background\n2. Key Components\n3. Relationships and Interactions\n4. Implications\n5. Conclusion"
            },
            
            "role_based": {
                "expert": "As an expert in the field, {task}",
                "teacher": "As an experienced teacher, explain {task} in a way that's easy to understand.",
                "consultant": "As a professional consultant, provide strategic advice on {task}"
            },
            
            "structured": {
                "report": "# Report: {task}\n\n## Executive Summary\n\n## Analysis\n\n## Recommendations\n\n## Conclusion",
                "comparison": "# Comparison: {task}\n\n## Option A\n### Pros\n### Cons\n\n## Option B\n### Pros\n### Cons\n\n## Recommendation",
                "plan": "# Plan: {task}\n\n## Objective\n\n## Steps\n1. \n2. \n3. \n\n## Timeline\n\n## Resources Needed"
            },
            
            "react": {
                "default": "Task: {task}\n\nI'll solve this using the following approach:\nThought: [reasoning]\nAction: [action to take]\nObservation: [result]\n...\nFinal Answer: [conclusion]",
                "research": "Research Task: {task}\n\nThought: I need to gather information about this topic\nAction: Search[...]\nObservation: [findings]\nThought: Based on this, I should...\nAction: ...\n",
                "problem_solving": "Problem: {task}\n\nThought: Let me break down this problem\nAction: Analyze[...]\nObservation: [analysis]\nThought: Now I can work on a solution\nAction: Solve[...]\n"
            }
        }
    
    async def get_template_by_task_type(self, task_type: str, strategy: str) -> Optional[str]:
        """Get the best template for a task type and strategy combination"""
        if not self.loaded:
            await self.load_templates()
            
        strategy_templates = self.templates.get(strategy, {})
        
        # Try to find specific task type template
        if task_type in strategy_templates:
            return strategy_templates[task_type]
            
        # Fall back to default
        return strategy_templates.get('default')