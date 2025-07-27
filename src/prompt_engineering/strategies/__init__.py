"""
Prompt engineering strategies for different use cases.
"""

from .zero_shot import ZeroShotStrategy
from .few_shot import FewShotStrategy
from .chain_of_thought import ChainOfThoughtStrategy
from .role_based import RoleBasedStrategy
from .structured import StructuredStrategy
from .react import ReActStrategy

__all__ = [
    'ZeroShotStrategy',
    'FewShotStrategy',
    'ChainOfThoughtStrategy',
    'RoleBasedStrategy',
    'StructuredStrategy',
    'ReActStrategy'
]