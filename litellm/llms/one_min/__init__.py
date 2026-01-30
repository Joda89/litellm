"""
1min.ai integration for LiteLLM

This module provides integration with 1min.ai's AI Feature API.
1min.ai provides a unified API endpoint for accessing multiple AI providers
including OpenAI, Anthropic, Google, Mistral, and more.

Official documentation: https://docs.1min.ai/docs/api/ai-feature-api

Example:
    >>> from litellm import completion
    >>> response = completion(
    ...     model="one_min/gpt-4o-mini",
    ...     messages=[{"role": "user", "content": "Hello"}]
    ... )
"""

from .config import (
    OneMinAIConfig,
    DEFAULT_CONFIG,
    FeatureType,
    ModelProvider,
    AVAILABLE_MODELS,
    IMAGE_MODELS,
    IMAGE_VARIATOR_MODELS,
    VIDEO_MODELS,
    TTS_MODELS,
)

from .exceptions import (
    OneMinAIError,
    OneMinAIAuthError,
    OneMinAIValidationError,
    OneMinAIAPIError,
    OneMinAIRateLimitError,
    OneMinAITimeoutError,
)

from .chat.handler import OneMinChatCompletion
from .chat.transformation import OneMinChatConfig

__all__ = [
    # Config
    "OneMinAIConfig",
    "DEFAULT_CONFIG",
    "FeatureType",
    "ModelProvider",
    "AVAILABLE_MODELS",
    "IMAGE_MODELS",
    "IMAGE_VARIATOR_MODELS",
    "VIDEO_MODELS",
    "TTS_MODELS",
    # Exceptions
    "OneMinAIError",
    "OneMinAIAuthError",
    "OneMinAIValidationError",
    "OneMinAIAPIError",
    "OneMinAIRateLimitError",
    "OneMinAITimeoutError",
    # Chat
    "OneMinChatCompletion",
    "OneMinChatConfig",
]
