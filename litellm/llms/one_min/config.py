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
    IMAGE_TO_PROMPT = "IMAGE_TO_PROMPT"

    # Code features
    CODE_GENERATOR = "CODE_GENERATOR"

    # Audio features
    TEXT_TO_SPEECH = "TEXT_TO_SPEECH"

    # Video features
    TEXT_TO_VIDEO = "TEXT_TO_VIDEO"

    # Writing features
    CONTENT_GENERATOR = "CONTENT_GENERATOR"

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
            cls.IMAGE_TO_PROMPT.value,
        ]

    @classmethod
    def is_audio_feature(cls, feature_type: str) -> bool:
        """Check if feature type is an audio feature"""
        return feature_type in [
            cls.TEXT_TO_SPEECH.value,
        ]

    @classmethod
    def is_video_feature(cls, feature_type: str) -> bool:
        """Check if feature type is a video feature"""
        return feature_type in [
            cls.TEXT_TO_VIDEO.value,
        ]

    @classmethod
    def is_writing_feature(cls, feature_type: str) -> bool:
        """Check if feature type is a writing feature"""
        return feature_type in [
            cls.CONTENT_GENERATOR.value,
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


# Available chat models by provider (from official documentation)
AVAILABLE_MODELS: Dict[str, List[str]] = {
    ModelProvider.OPENAI.value: [
        "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo",
        "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano",
        "o3", "o3-mini", "o3-pro",
    ],
    ModelProvider.ANTHROPIC.value: [
        "claude-sonnet-4-5-20250929", "claude-sonnet-4-20250514",
        "claude-opus-4-5-20251101", "claude-haiku-4-5-20251001",
    ],
    ModelProvider.GOOGLE.value: [
        "gemini-2.5-pro", "gemini-2.5-flash",
    ],
    ModelProvider.MISTRAL.value: [
        "mistral-small-latest", "mistral-medium-latest",
        "mistral-large-latest", "open-mistral-nemo",
    ],
    ModelProvider.META.value: [
        "meta/meta-llama-3.1-405b-instruct", "meta/meta-llama-3-70b-instruct",
    ],
    ModelProvider.DEEPSEEK.value: [
        "deepseek-reasoner", "deepseek-chat",
    ],
    ModelProvider.PERPLEXITY.value: [
        "sonar-reasoning-pro", "sonar-reasoning", "sonar-pro", "sonar",
    ],
    ModelProvider.XAI.value: [
        "grok-3", "grok-3-mini",
    ],
    ModelProvider.ALIBABA.value: [
        "qwen3-max", "qwen-plus", "qwen-max", "qwen-flash",
    ],
    ModelProvider.COHERE.value: [
        "command-r-08-2024",
    ],
}

# Image generation models
IMAGE_MODELS = [
    # Magic Art
    "magic-art-5.2", "magic-art-6.1", "magic-art-7.0",
    # OpenAI
    "gpt-image-1", "gpt-image-1-mini", "dall-e-3", "dall-e-2",
    # Leonardo AI
    "leonardo-phoenix", "leonardo-lightning-xl", "leonardo-anime-xl",
    "leonardo-diffusion-xl", "leonardo-kino-xl", "leonardo-vision-xl",
    "leonardo-albedo-base-xl",
    # Stability AI
    "stable-diffusion-xl-1.0", "stable-image-core", "stable-image-ultra",
    # Flux
    "flux-pro", "flux-krea-dev", "flux-dev", "flux-schnell",
    "flux-schnell-lora", "flux-dev-lora", "flux-pro-1.1", "flux-1.1-pro-ultra",
    # Google
    "gemini-2.5-flash-image", "gemini-3-pro-image-preview",
    # Autres
    "dzine", "grok-2-image", "qwen-image", "recraft",
]

# Image variation models
IMAGE_VARIATOR_MODELS = [
    "dall-e-2", "clipdrop", "dzine",
    "magic-art-5.2", "magic-art-6.1", "magic-art-7.0",
    "flux-redux-dev", "flux-redux-schnell", "recraft",
]

# Video generation models
VIDEO_MODELS = [
    "tongyi", "kling-ai", "luma-ai", "veo3-video",
    "hunyuan-ai", "wanx-ai", "hailuo-ai", "pika-ai", "animate-diff",
]

# Text-to-speech models
TTS_MODELS = ["tts-1", "tts-1-hd"]

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
        "claude-haiku-4-5-20251001",
        # DeepSeek
        "deepseek-reasoner",
        "deepseek-chat",
        # OpenAI
        "gpt-4o",
        "o3",
        # xAI
        "grok-code-fast-1",
    ],
    FeatureType.IMAGE_GENERATOR.value: IMAGE_MODELS,
    FeatureType.IMAGE_VARIATOR.value: IMAGE_VARIATOR_MODELS,
    FeatureType.TEXT_TO_VIDEO.value: VIDEO_MODELS,
    FeatureType.TEXT_TO_SPEECH.value: TTS_MODELS,
    FeatureType.IMAGE_TO_PROMPT.value: IMAGE_MODELS,
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
    return (
        model in all_models
        or model in IMAGE_MODELS
        or model in IMAGE_VARIATOR_MODELS
        or model in VIDEO_MODELS
        or model in TTS_MODELS
    )


def is_model_supported_for_feature(model: str, feature_type: str) -> bool:
    """
    Check if a model supports a specific feature type

    Rules:
    - If feature has an allowlist in FEATURE_SUPPORTED_MODELS, enforce it.
    - All other features: allowed unless restricted
    """
    if feature_type in FEATURE_SUPPORTED_MODELS:
        return model in FEATURE_SUPPORTED_MODELS[feature_type]

    # All other features are supported by all models
    return True


def get_supported_models_for_feature(feature_type: str) -> List[str]:
    """Get all models that support a specific feature type"""
    if feature_type in FEATURE_SUPPORTED_MODELS:
        return FEATURE_SUPPORTED_MODELS[feature_type]

    return get_all_models()
