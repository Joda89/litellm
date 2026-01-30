"""
Tests for 1min.ai model configuration (config.py)
"""
import pytest

from litellm.llms.one_min.config import (
    FeatureType,
    ModelProvider,
    AVAILABLE_MODELS,
    IMAGE_MODELS,
    IMAGE_VARIATOR_MODELS,
    VIDEO_MODELS,
    TTS_MODELS,
    FEATURE_SUPPORTED_MODELS,
    CODE_MODELS,
    DEFAULT_CONFIG,
    OneMinAIConfig,
    get_all_models,
    get_models_by_provider,
    is_valid_model,
    is_model_supported_for_feature,
    get_supported_models_for_feature,
)


class TestFeatureType:
    """Test FeatureType enum"""

    def test_chat_features_exist(self):
        assert FeatureType.CHAT_WITH_AI.value == "CHAT_WITH_AI"
        assert FeatureType.CHAT_WITH_IMAGE.value == "CHAT_WITH_IMAGE"
        assert FeatureType.CHAT_WITH_PDF.value == "CHAT_WITH_PDF"
        assert FeatureType.CHAT_WITH_YOUTUBE_VIDEO.value == "CHAT_WITH_YOUTUBE_VIDEO"

    def test_image_features_exist(self):
        assert FeatureType.IMAGE_GENERATOR.value == "IMAGE_GENERATOR"
        assert FeatureType.IMAGE_VARIATOR.value == "IMAGE_VARIATOR"
        assert FeatureType.IMAGE_TO_PROMPT.value == "IMAGE_TO_PROMPT"

    def test_code_feature_exists(self):
        assert FeatureType.CODE_GENERATOR.value == "CODE_GENERATOR"

    def test_audio_feature_exists(self):
        assert FeatureType.TEXT_TO_SPEECH.value == "TEXT_TO_SPEECH"

    def test_video_feature_exists(self):
        assert FeatureType.TEXT_TO_VIDEO.value == "TEXT_TO_VIDEO"

    def test_writing_feature_exists(self):
        assert FeatureType.CONTENT_GENERATOR.value == "CONTENT_GENERATOR"

    def test_removed_features_do_not_exist(self):
        """Verify that non-documented feature types were removed"""
        feature_names = [f.name for f in FeatureType]
        assert "CODE_EXPLAINER" not in feature_names
        assert "CODE_OPTIMIZER" not in feature_names
        assert "TEXT_SUMMARIZER" not in feature_names
        assert "TEXT_TRANSLATOR" not in feature_names
        assert "TEXT_WRITER" not in feature_names
        assert "TEXT_REWRITER" not in feature_names
        assert "SPEECH_TO_TEXT" not in feature_names
        assert "SENTIMENT_ANALYZER" not in feature_names
        assert "CONTENT_MODERATOR" not in feature_names
        assert "WEB_SEARCH" not in feature_names
        assert "IMAGE_SEARCH" not in feature_names

    def test_total_feature_count(self):
        assert len(list(FeatureType)) == 11


class TestFeatureTypeClassification:
    """Test feature type classification methods"""

    def test_is_chat_feature(self):
        assert FeatureType.is_chat_feature("CHAT_WITH_AI") is True
        assert FeatureType.is_chat_feature("CHAT_WITH_IMAGE") is True
        assert FeatureType.is_chat_feature("CHAT_WITH_PDF") is True
        assert FeatureType.is_chat_feature("CHAT_WITH_YOUTUBE_VIDEO") is True
        assert FeatureType.is_chat_feature("IMAGE_GENERATOR") is False

    def test_is_image_feature(self):
        assert FeatureType.is_image_feature("IMAGE_GENERATOR") is True
        assert FeatureType.is_image_feature("IMAGE_VARIATOR") is True
        assert FeatureType.is_image_feature("IMAGE_TO_PROMPT") is True
        assert FeatureType.is_image_feature("CHAT_WITH_AI") is False

    def test_is_audio_feature(self):
        assert FeatureType.is_audio_feature("TEXT_TO_SPEECH") is True
        assert FeatureType.is_audio_feature("CHAT_WITH_AI") is False

    def test_is_video_feature(self):
        assert FeatureType.is_video_feature("TEXT_TO_VIDEO") is True
        assert FeatureType.is_video_feature("CHAT_WITH_AI") is False

    def test_is_writing_feature(self):
        assert FeatureType.is_writing_feature("CONTENT_GENERATOR") is True
        assert FeatureType.is_writing_feature("CHAT_WITH_AI") is False

    def test_requires_conversation_id(self):
        assert FeatureType.requires_conversation_id("CHAT_WITH_PDF") is True
        assert FeatureType.requires_conversation_id("CHAT_WITH_YOUTUBE_VIDEO") is True
        assert FeatureType.requires_conversation_id("CHAT_WITH_AI") is False

    def test_removed_classification_methods(self):
        """Verify removed classification methods no longer exist"""
        assert not hasattr(FeatureType, "is_code_feature")
        assert not hasattr(FeatureType, "is_text_feature")
        assert not hasattr(FeatureType, "is_analysis_feature")
        assert not hasattr(FeatureType, "is_search_feature")


class TestModelProvider:
    """Test ModelProvider enum"""

    def test_all_providers_exist(self):
        assert len(list(ModelProvider)) == 10
        assert ModelProvider.OPENAI.value == "openai"
        assert ModelProvider.ANTHROPIC.value == "anthropic"
        assert ModelProvider.GOOGLE.value == "google"


class TestAvailableModels:
    """Test AVAILABLE_MODELS dictionary"""

    def test_no_extra_category(self):
        assert "extra" not in AVAILABLE_MODELS

    def test_all_providers_have_models(self):
        for provider in ModelProvider:
            assert provider.value in AVAILABLE_MODELS
            assert len(AVAILABLE_MODELS[provider.value]) > 0

    def test_openai_models_cleaned(self):
        openai_models = AVAILABLE_MODELS["openai"]
        assert "gpt-4o" in openai_models
        assert "gpt-4o-mini" in openai_models
        assert "o3" in openai_models
        # Removed models
        for m in openai_models:
            assert not m.startswith("gpt-5")
            assert not m.startswith("o4-")

    def test_anthropic_models_cleaned(self):
        anthropic_models = AVAILABLE_MODELS["anthropic"]
        assert "claude-opus-4-5-20251101" in anthropic_models
        assert "claude-opus-4-20250514" not in anthropic_models
        assert "claude-opus-4-1-20250805" not in anthropic_models

    def test_google_models_cleaned(self):
        google_models = AVAILABLE_MODELS["google"]
        assert "gemini-2.5-pro" in google_models
        assert "gemini-3-pro-preview" not in google_models

    def test_xai_models_cleaned(self):
        xai_models = AVAILABLE_MODELS["xai"]
        assert "grok-3" in xai_models
        for m in xai_models:
            assert not m.startswith("grok-4")


class TestImageModels:
    """Test IMAGE_MODELS list"""

    def test_contains_major_models(self):
        assert "dall-e-3" in IMAGE_MODELS
        assert "flux-pro" in IMAGE_MODELS
        assert "stable-diffusion-xl-1.0" in IMAGE_MODELS
        assert "magic-art-7.0" in IMAGE_MODELS

    def test_old_models_removed(self):
        assert "magic-art" not in IMAGE_MODELS
        assert "black-forest-labs/flux-schnell" not in IMAGE_MODELS


class TestImageVariatorModels:
    """Test IMAGE_VARIATOR_MODELS list"""

    def test_contains_expected_models(self):
        assert "dall-e-2" in IMAGE_VARIATOR_MODELS
        assert "clipdrop" in IMAGE_VARIATOR_MODELS
        assert "recraft" in IMAGE_VARIATOR_MODELS


class TestVideoModels:
    """Test VIDEO_MODELS list"""

    def test_contains_expected_models(self):
        assert "kling-ai" in VIDEO_MODELS
        assert "luma-ai" in VIDEO_MODELS
        assert "veo3-video" in VIDEO_MODELS
        assert len(VIDEO_MODELS) == 9


class TestTTSModels:
    """Test TTS_MODELS list"""

    def test_contains_expected_models(self):
        assert "tts-1" in TTS_MODELS
        assert "tts-1-hd" in TTS_MODELS
        assert len(TTS_MODELS) == 2


class TestFeatureSupportedModels:
    """Test FEATURE_SUPPORTED_MODELS mappings"""

    def test_code_generator_has_allowlist(self):
        assert FeatureType.CODE_GENERATOR.value in FEATURE_SUPPORTED_MODELS
        models = FEATURE_SUPPORTED_MODELS[FeatureType.CODE_GENERATOR.value]
        assert "gpt-4o" in models
        assert "deepseek-chat" in models

    def test_image_generator_maps_to_image_models(self):
        assert FEATURE_SUPPORTED_MODELS[FeatureType.IMAGE_GENERATOR.value] is IMAGE_MODELS

    def test_image_variator_maps_to_variator_models(self):
        assert FEATURE_SUPPORTED_MODELS[FeatureType.IMAGE_VARIATOR.value] is IMAGE_VARIATOR_MODELS

    def test_video_maps_to_video_models(self):
        assert FEATURE_SUPPORTED_MODELS[FeatureType.TEXT_TO_VIDEO.value] is VIDEO_MODELS

    def test_tts_maps_to_tts_models(self):
        assert FEATURE_SUPPORTED_MODELS[FeatureType.TEXT_TO_SPEECH.value] is TTS_MODELS

    def test_image_to_prompt_maps_to_image_models(self):
        assert FEATURE_SUPPORTED_MODELS[FeatureType.IMAGE_TO_PROMPT.value] is IMAGE_MODELS


class TestModelValidation:
    """Test model validation functions"""

    def test_get_all_models(self):
        models = get_all_models()
        assert "gpt-4o" in models
        assert "deepseek-chat" in models
        assert len(models) > 0

    def test_get_models_by_provider(self):
        models = get_models_by_provider("openai")
        assert "gpt-4o" in models

    def test_get_models_by_provider_invalid(self):
        assert get_models_by_provider("nonexistent") == []

    def test_is_valid_model_chat(self):
        assert is_valid_model("gpt-4o") is True

    def test_is_valid_model_image(self):
        assert is_valid_model("dall-e-3") is True

    def test_is_valid_model_video(self):
        assert is_valid_model("kling-ai") is True

    def test_is_valid_model_tts(self):
        assert is_valid_model("tts-1") is True

    def test_is_valid_model_invalid(self):
        assert is_valid_model("nonexistent-model") is False

    def test_is_model_supported_for_feature_with_allowlist(self):
        assert is_model_supported_for_feature("gpt-4o", "CODE_GENERATOR") is True
        assert is_model_supported_for_feature("nonexistent", "CODE_GENERATOR") is False

    def test_is_model_supported_for_feature_without_allowlist(self):
        assert is_model_supported_for_feature("gpt-4o", "CHAT_WITH_AI") is True

    def test_get_supported_models_for_feature_with_allowlist(self):
        models = get_supported_models_for_feature("CODE_GENERATOR")
        assert "gpt-4o" in models

    def test_get_supported_models_for_feature_without_allowlist(self):
        models = get_supported_models_for_feature("CHAT_WITH_AI")
        assert "gpt-4o" in models


class TestDefaultConfig:
    """Test DEFAULT_CONFIG"""

    def test_api_base(self):
        assert DEFAULT_CONFIG["api_base"] == "https://api.1min.ai/api/features"

    def test_timeout(self):
        assert DEFAULT_CONFIG["timeout"] == 60
