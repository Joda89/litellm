"""
Tests for 1min.ai models in config.py
"""
import pytest

from litellm.llms.one_min.config import (
    AVAILABLE_MODELS,
    IMAGE_MODELS,
    IMAGE_VARIATOR_MODELS,
    VIDEO_MODELS,
    TTS_MODELS,
    FEATURE_SUPPORTED_MODELS,
    CODE_MODELS,
    ModelProvider,
    FeatureType,
    get_all_models,
    is_valid_model,
    is_model_supported_for_feature,
)


class TestChatModels:
    """Test chat model lists per provider"""

    def test_openai_models(self):
        models = AVAILABLE_MODELS[ModelProvider.OPENAI.value]
        expected = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo",
                     "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano",
                     "o3", "o3-mini", "o3-pro"]
        assert models == expected

    def test_anthropic_models(self):
        models = AVAILABLE_MODELS[ModelProvider.ANTHROPIC.value]
        assert len(models) == 4
        for m in models:
            assert m.startswith("claude-")

    def test_deepseek_models(self):
        models = AVAILABLE_MODELS[ModelProvider.DEEPSEEK.value]
        assert models == ["deepseek-reasoner", "deepseek-chat"]

    def test_perplexity_models(self):
        models = AVAILABLE_MODELS[ModelProvider.PERPLEXITY.value]
        assert "sonar-deep-research" not in models
        assert "sonar" in models

    def test_all_models_are_strings(self):
        for provider, models in AVAILABLE_MODELS.items():
            for model in models:
                assert isinstance(model, str)
                assert len(model) > 0


class TestSpecializedModelLists:
    """Test specialized model lists"""

    def test_image_models_are_not_empty(self):
        assert len(IMAGE_MODELS) > 0

    def test_image_variator_models_are_not_empty(self):
        assert len(IMAGE_VARIATOR_MODELS) > 0

    def test_video_models_are_not_empty(self):
        assert len(VIDEO_MODELS) > 0

    def test_tts_models_are_not_empty(self):
        assert len(TTS_MODELS) > 0

    def test_code_models_match_feature_supported(self):
        assert CODE_MODELS == list(FEATURE_SUPPORTED_MODELS[FeatureType.CODE_GENERATOR.value])

    def test_no_overlap_chat_and_image(self):
        """Chat models and image models should not overlap"""
        chat_models = set(get_all_models())
        image_set = set(IMAGE_MODELS)
        overlap = chat_models & image_set
        assert len(overlap) == 0, f"Overlapping models: {overlap}"

    def test_no_overlap_chat_and_video(self):
        """Chat models and video models should not overlap"""
        chat_models = set(get_all_models())
        video_set = set(VIDEO_MODELS)
        overlap = chat_models & video_set
        assert len(overlap) == 0, f"Overlapping models: {overlap}"


class TestModelFeatureSupport:
    """Test model-feature support relationships"""

    def test_image_model_supported_for_image_generator(self):
        assert is_model_supported_for_feature("dall-e-3", "IMAGE_GENERATOR") is True

    def test_chat_model_not_supported_for_image_generator(self):
        assert is_model_supported_for_feature("gpt-4o", "IMAGE_GENERATOR") is False

    def test_video_model_supported_for_text_to_video(self):
        assert is_model_supported_for_feature("kling-ai", "TEXT_TO_VIDEO") is True

    def test_tts_model_supported_for_text_to_speech(self):
        assert is_model_supported_for_feature("tts-1", "TEXT_TO_SPEECH") is True

    def test_chat_model_supported_for_chat(self):
        assert is_model_supported_for_feature("gpt-4o", "CHAT_WITH_AI") is True
