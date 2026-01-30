"""
Chat module for 1min.ai integration
"""

from .handler import OneMinChatCompletion
from .transformation import OneMinChatConfig

__all__ = [
    "OneMinChatCompletion",
    "OneMinChatConfig",
]
