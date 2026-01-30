"""
Configuration and constants for 1min.ai integration

Based on official documentation: https://docs.1min.ai/docs/api/ai-feature-api
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


class FeatureType(str, Enum):
    """
    Available 1min.ai feature types

    Based on: https://docs.1min.ai/docs/api/ai-feature-api
    """
    CHAT_WITH_AI = "CHAT_WITH_AI"
    CHAT_WITH_IMAGE = "CHAT_WITH_IMAGE"
    CHAT_WITH_PDF = "CHAT_WITH_PDF"
    CHAT_WITH_YOUTUBE_VIDEO = "CHAT_WITH_YOUTUBE_VIDEO"
    IMAGE_GENERATOR = "IMAGE_GENERATOR"
    IMAGE_VARIATOR = "IMAGE_VARIATOR"

    @classmethod
    def is_chat_feature(cls, feature_type: str) -> bool:
        """Check if feature type is a chat feature"""
        return feature_type in [
            cls.CHAT_WITH_AI.value,
            cls.CHAT_WITH_IMAGE.value,
            cls.CHAT_WITH_PDF.value,
            cls.CHAT_WITH_YOUTUBE_VIDEO.value
        ]

    @classmethod
    def is_image_feature(cls, feature_type: str) -> bool:
        """Check if feature type is an image feature"""
        return feature_type in [
            cls.IMAGE_GENERATOR.value,
            cls.IMAGE_VARIATOR.value
        ]

    @classmethod
    def requires_conversation_id(cls, feature_type: str) -> bool:
        """Check if feature type requires a conversation ID"""
        return feature_type in [
            cls.CHAT_WITH_PDF.value,
            cls.CHAT_WITH_YOUTUBE_VIDEO.value
        ]


class ModelProvider(str, Enum):
    """Model providers available through 1min.ai"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    MISTRAL = "mistral"
    META = "meta"
    DEEPSEEK = "deepseek"
    PERPLEXITY = "perplexity"
    XAI = "xai"
    ALIBABA = "alibaba"
    COHERE = "cohere"


# Available models by provider (from official documentation)
AVAILABLE_MODELS: Dict[str, List[str]] = {
    ModelProvider.OPENAI.value: [
        "gpt-5.2-pro", "gpt-5.2", "gpt-5.1", "gpt-5", "gpt-5-mini", "gpt-5-nano",
        "gpt-5-chat-latest", "gpt-5.1-codex", "gpt-5.1-codex-mini",
        "gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano",
        "gpt-4-turbo", "gpt-3.5-turbo",
        "o4-mini", "o3-mini", "o3", "o3-pro", "o3-deep-research",
        "o4-mini-deep-research",
    ],
    ModelProvider.ANTHROPIC.value: [
        "claude-sonnet-4-5-20250929", "claude-sonnet-4-20250514",
        "claude-opus-4-5-20251101", "claude-opus-4-20250514", "claude-opus-4-1-20250805",
        "claude-haiku-4-5-20251001",
    ],
    ModelProvider.GOOGLE.value: [
        "gemini-3-pro-preview", "gemini-2.5-pro", "gemini-2.5-flash",
    ],
    ModelProvider.MISTRAL.value: [
        "magistral-small-latest", "magistral-medium-latest",
        "ministral-14b-latest", "open-mistral-nemo",
        "mistral-small-latest", "mistral-medium-latest", "mistral-large-latest",
    ],
    ModelProvider.META.value: [
        "meta/meta-llama-3.1-405b-instruct", "meta/meta-llama-3-70b-instruct",
        "meta/llama-4-scout-instruct", "meta/llama-4-maverick-instruct",
        "meta/llama-2-70b-chat",
    ],
    ModelProvider.DEEPSEEK.value: [
        "deepseek-reasoner", "deepseek-chat",
    ],
    ModelProvider.PERPLEXITY.value: [
        "sonar-reasoning-pro", "sonar-reasoning", "sonar-pro",
        "sonar-deep-research", "sonar",
    ],
    ModelProvider.XAI.value: [
        "grok-4-fast-reasoning", "grok-4-fast-non-reasoning",
        "grok-4-0709", "grok-3-mini", "grok-3",
    ],
    ModelProvider.ALIBABA.value: [
        "qwen3-max", "qwen-plus", "qwen-max", "qwen-flash",
    ],
    ModelProvider.COHERE.value: [
        "command-r-08-2024",
    ],
    "extra": [
        "openai/gpt-oss-20b", "openai/gpt-oss-120b",
    ],
}

# Image generation models
IMAGE_MODELS = [
    "flux-pro", "flux-dev", "black-forest-labs/flux-schnell", "magic-art"
]


# Default configuration
DEFAULT_CONFIG = {
    "api_base": "https://api.1min.ai/api/features",
    "timeout": 60,
    "max_retries": 3,
    "retry_delay": 1.0,
    "web_search_num_sites": 1,
    "web_search_max_words": 500,
    "image_aspect_ratio": "1:1",
    "image_output_format": "webp",
    "image_num_outputs": 1,
}


@dataclass
class OneMinAIConfig:
    """Configuration for 1min.ai handler"""
    api_key: str
    api_base: str = DEFAULT_CONFIG["api_base"]
    timeout: int = DEFAULT_CONFIG["timeout"]
    max_retries: int = DEFAULT_CONFIG["max_retries"]
    retry_delay: float = DEFAULT_CONFIG["retry_delay"]

    def __post_init__(self):
        from .exceptions import OneMinAIAuthError
        if not self.api_key:
            raise OneMinAIAuthError("API key is required")


def get_all_models() -> List[str]:
    """Get a flat list of all available models"""
    models = []
    for provider_models in AVAILABLE_MODELS.values():
        models.extend(provider_models)
    return models


def get_models_by_provider(provider: str) -> List[str]:
    """Get models for a specific provider"""
    return AVAILABLE_MODELS.get(provider, [])


def is_valid_model(model: str) -> bool:
    """Check if a model is valid"""
    all_models = get_all_models()
    return model in all_models or model in IMAGE_MODELS
