"""
Tests for 1min.ai models configuration
"""
import os
import sys
import pytest

sys.path.insert(0, os.path.abspath("../../../../../.."))

from litellm.llms.one_min.models import (
    OneMinModelType,
    DEFAULT_MODELS,
    AVAILABLE_MODELS,
    get_model_type,
    is_valid_model,
    get_model_info,
    get_models_by_type,
    get_all_models,
    get_all_model_types,
)


class TestOneMinModelType:
    """Test OneMinModelType enumeration"""

    def test_model_types_exist(self):
        """Test that all model types are defined"""
        assert hasattr(OneMinModelType, "CODE_GENERATOR")
        assert hasattr(OneMinModelType, "CODE_EXPLAINER")
        assert hasattr(OneMinModelType, "CODE_TRANSLATOR")
        assert hasattr(OneMinModelType, "CODE_REVIEWER")
        assert hasattr(OneMinModelType, "IMAGE_GENERATOR")
        assert hasattr(OneMinModelType, "TEXT_COMPLETION")
        assert hasattr(OneMinModelType, "CHAT")
        assert hasattr(OneMinModelType, "DOCUMENT_PROCESSOR")

    def test_model_type_values(self):
        """Test that model types have correct values"""
        assert OneMinModelType.CODE_GENERATOR.value == "code-generator"
        assert OneMinModelType.CODE_EXPLAINER.value == "code-explainer"
        assert OneMinModelType.CODE_TRANSLATOR.value == "code-translator"
        assert OneMinModelType.CODE_REVIEWER.value == "code-reviewer"
        assert OneMinModelType.IMAGE_GENERATOR.value == "image-generator"
        assert OneMinModelType.TEXT_COMPLETION.value == "text-completion"
        assert OneMinModelType.CHAT.value == "chat"
        assert OneMinModelType.DOCUMENT_PROCESSOR.value == "document-processor"

    def test_model_type_count(self):
        """Test that all 8 model types exist"""
        model_types = list(OneMinModelType)
        assert len(model_types) == 8


class TestDefaultModels:
    """Test default models configuration"""

    def test_default_models_exist(self):
        """Test that default models are defined for each type"""
        assert len(DEFAULT_MODELS) == 8

    def test_default_models_format(self):
        """Test that default models have correct format"""
        for model_type, model_name in DEFAULT_MODELS.items():
            assert isinstance(model_type, OneMinModelType)
            assert model_name.startswith("one_min/")
            assert isinstance(model_name, str)

    def test_default_models_values(self):
        """Test specific default model values"""
        assert DEFAULT_MODELS[OneMinModelType.CODE_GENERATOR] == "one_min/code-generator"
        assert DEFAULT_MODELS[OneMinModelType.CHAT] == "one_min/chat"
        assert DEFAULT_MODELS[OneMinModelType.IMAGE_GENERATOR] == "one_min/image-generator"


class TestAvailableModels:
    """Test available models configuration"""

    def test_available_models_count(self):
        """Test that all 8 models are listed"""
        assert len(AVAILABLE_MODELS) == 8

    def test_available_models_structure(self):
        """Test that each model has required fields"""
        for model_name, model_info in AVAILABLE_MODELS.items():
            assert "type" in model_info
            assert "description" in model_info
            assert "use_cases" in model_info
            assert "input_type" in model_info
            assert "output_type" in model_info

    def test_model_info_completeness(self):
        """Test that model info is complete"""
        for model_name, model_info in AVAILABLE_MODELS.items():
            # Check that type matches model name
            assert model_name.startswith("one_min/")

            # Check that all fields are non-empty
            assert len(model_info["type"]) > 0
            assert len(model_info["description"]) > 0
            assert len(model_info["use_cases"]) > 0


class TestModelTypeFunctions:
    """Test model type utility functions"""

    def test_get_model_type(self):
        """Test get_model_type function"""
        assert get_model_type("one_min/code-generator") == "CODE_GENERATOR"
        assert get_model_type("one_min/chat") == "CHAT"
        assert get_model_type("one_min/image-generator") == "IMAGE_GENERATOR"

    def test_get_model_type_invalid(self):
        """Test get_model_type with invalid model"""
        assert get_model_type("invalid-model") == "UNKNOWN"
        assert get_model_type("") == "UNKNOWN"

    def test_is_valid_model(self):
        """Test is_valid_model function"""
        assert is_valid_model("one_min/code-generator") is True
        assert is_valid_model("one_min/chat") is True
        assert is_valid_model("invalid-model") is False
        assert is_valid_model("gpt-4") is False

    def test_get_model_info(self):
        """Test get_model_info function"""
        info = get_model_info("one_min/code-generator")
        assert info is not None
        assert info["type"] == "CODE_GENERATOR"
        assert "description" in info

    def test_get_model_info_invalid(self):
        """Test get_model_info with invalid model"""
        info = get_model_info("invalid-model")
        assert info is None

    def test_get_models_by_type(self):
        """Test get_models_by_type function"""
        code_models = get_models_by_type("CODE_GENERATOR")
        assert "one_min/code-generator" in code_models

        chat_models = get_models_by_type("CHAT")
        assert "one_min/chat" in chat_models

    def test_get_models_by_type_multiple(self):
        """Test that get_models_by_type returns all models of a type"""
        all_models = []
        for model_type in OneMinModelType:
            models = get_models_by_type(model_type.name)
            all_models.extend(models)

        assert len(all_models) == len(AVAILABLE_MODELS)

    def test_get_all_models(self):
        """Test get_all_models function"""
        models = get_all_models()
        assert len(models) == 8
        assert "one_min/code-generator" in models
        assert "one_min/chat" in models
        assert "one_min/image-generator" in models

    def test_get_all_model_types(self):
        """Test get_all_model_types function"""
        types = get_all_model_types()
        assert len(types) == 8
        assert "code-generator" in types
        assert "chat" in types
        assert "image-generator" in types


class TestModelCoverage:
    """Test that all models are properly covered"""

    def test_all_models_have_info(self):
        """Test that all available models have complete info"""
        for model_name in get_all_models():
            info = get_model_info(model_name)
            assert info is not None, f"Missing info for {model_name}"

    def test_all_types_have_models(self):
        """Test that all model types have at least one model"""
        for model_type in OneMinModelType:
            models = get_models_by_type(model_type.name)
            assert len(models) > 0, f"No models for type {model_type.name}"

    def test_default_model_in_available(self):
        """Test that all default models are in available models"""
        for model_type, default_model in DEFAULT_MODELS.items():
            assert is_valid_model(default_model)
            assert get_model_info(default_model) is not None
