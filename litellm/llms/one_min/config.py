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
    # Chat features
    CHAT_WITH_AI = "CHAT_WITH_AI"
    CHAT_WITH_IMAGE = "CHAT_WITH_IMAGE"
    CHAT_WITH_PDF = "CHAT_WITH_PDF"
    CHAT_WITH_YOUTUBE_VIDEO = "CHAT_WITH_YOUTUBE_VIDEO"

    # Image features
    IMAGE_GENERATOR = "IMAGE_GENERATOR"
    IMAGE_VARIATOR = "IMAGE_VARIATOR"

    # Code features
    CODE_GENERATOR = "CODE_GENERATOR"
    CODE_EXPLAINER = "CODE_EXPLAINER"
    CODE_OPTIMIZER = "CODE_OPTIMIZER"

    # Text features
    TEXT_SUMMARIZER = "TEXT_SUMMARIZER"
    TEXT_TRANSLATOR = "TEXT_TRANSLATOR"
    TEXT_WRITER = "TEXT_WRITER"
    TEXT_REWRITER = "TEXT_REWRITER"

    # Audio features
    SPEECH_TO_TEXT = "SPEECH_TO_TEXT"
    TEXT_TO_SPEECH = "TEXT_TO_SPEECH"

    # Analysis features
    SENTIMENT_ANALYZER = "SENTIMENT_ANALYZER"
    CONTENT_MODERATOR = "CONTENT_MODERATOR"

    # Search features
    WEB_SEARCH = "WEB_SEARCH"
    IMAGE_SEARCH = "IMAGE_SEARCH"

    @classmethod
    def is_chat_feature(cls, feature_type: str) -> bool:
        """Check if feature type is a chat feature"""
        return feature_type in [
            cls.CHAT_WITH_AI.value,
            cls.CHAT_WITH_IMAGE.value,
            cls.CHAT_WITH_PDF.value,
            cls.CHAT_WITH_YOUTUBE_VIDEO.value,
        ]

    @classmethod
    def is_image_feature(cls, feature_type: str) -> bool:
        """Check if feature type is an image feature"""
        return feature_type in [
            cls.IMAGE_GENERATOR.value,
            cls.IMAGE_VARIATOR.value,
            cls.IMAGE_SEARCH.value,
        ]

    @classmethod
    def is_code_feature(cls, feature_type: str) -> bool:
        """Check if feature type is a code feature"""
        return feature_type in [
            cls.CODE_GENERATOR.value,
            cls.CODE_EXPLAINER.value,
            cls.CODE_OPTIMIZER.value,
        ]

    @classmethod
    def is_text_feature(cls, feature_type: str) -> bool:
        """Check if feature type is a text feature"""
        return feature_type in [
            cls.TEXT_SUMMARIZER.value,
            cls.TEXT_TRANSLATOR.value,
            cls.TEXT_WRITER.value,
            cls.TEXT_REWRITER.value,
        ]

    @classmethod
    def is_audio_feature(cls, feature_type: str) -> bool:
        """Check if feature type is an audio feature"""
        return feature_type in [
            cls.SPEECH_TO_TEXT.value,
            cls.TEXT_TO_SPEECH.value,
        ]

    @classmethod
    def is_analysis_feature(cls, feature_type: str) -> bool:
        """Check if feature type is an analysis feature"""
        return feature_type in [
            cls.SENTIMENT_ANALYZER.value,
            cls.CONTENT_MODERATOR.value,
        ]

    @classmethod
    def is_search_feature(cls, feature_type: str) -> bool:
        """Check if feature type is a search feature"""
        return feature_type in [
            cls.WEB_SEARCH.value,
            cls.IMAGE_SEARCH.value,
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

# Feature-specific model allowlists (from 1min.ai docs when provided)
FEATURE_SUPPORTED_MODELS: Dict[str, List[str]] = {
    FeatureType.CODE_GENERATOR.value: [
        # Alibaba
        "qwen3-coder-plus",
        "qwen3-coder-flash",
        # Anthropic
        "claude-sonnet-4-5-20250929",
        "claude-sonnet-4-20250514",
        "claude-opus-4-5-20251101",
        "claude-opus-4-1-20250805",
        "claude-haiku-4-5-20251001",
        # DeepSeek
        "deepseek-reasoner",
        "deepseek-chat",
        # GoogleAI
        "gemini-3-pro-preview",
        # OpenAI
        "gpt-5.1-codex-mini",
        "gpt-5.1-codex",
        "gpt-5-chat-latest",
        "gpt-5",
        "gpt-4o",
        "o3",
        # xAI
        "grok-code-fast-1",
    ]
}

# Code generation models (used for CODE_GENERATOR feature)
CODE_MODELS = list(FEATURE_SUPPORTED_MODELS.get(FeatureType.CODE_GENERATOR.value, []))


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


def is_model_supported_for_feature(model: str, feature_type: str) -> bool:
    """
    Check if a model supports a specific feature type

    Rules:
    - If feature has an allowlist in FEATURE_SUPPORTED_MODELS, enforce it.
    - Image features: Only specific models (IMAGE_MODELS)
    - All other features: allowed unless restricted
    """
    # Enforce allowlist when provided
    if feature_type in FEATURE_SUPPORTED_MODELS:
        return model in FEATURE_SUPPORTED_MODELS[feature_type]

    # Image generation only with specific models
    if FeatureType.is_image_feature(feature_type):
        return model in IMAGE_MODELS

    # All other features are supported by all models
    return True


def get_supported_models_for_feature(feature_type: str) -> List[str]:
    """Get all models that support a specific feature type"""
    if feature_type in FEATURE_SUPPORTED_MODELS:
        return FEATURE_SUPPORTED_MODELS[feature_type]

    if FeatureType.is_image_feature(feature_type):
        return IMAGE_MODELS

    return get_all_models()

