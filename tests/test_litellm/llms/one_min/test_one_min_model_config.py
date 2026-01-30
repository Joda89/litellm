"""
Tests for 1min.ai model configuration
"""
import os
import sys
import pytest

sys.path.insert(0, os.path.abspath("../../../../../.."))

from litellm.llms.one_min.model_config import (
    ModelTypeConfig,
    MODEL_TYPE_CONFIGS,
    get_model_type_config,
    is_streaming_supported,
    is_tool_calling_supported,
    get_supported_parameters,
    get_default_parameters,
    get_max_tokens,
    get_context_window,
)


class TestModelTypeConfig:
    """Test ModelTypeConfig dataclass"""

    def test_config_initialization(self):
        """Test that ModelTypeConfig can be initialized"""
        config = ModelTypeConfig(
            name="Test Model",
            description="Test description",
            supported_parameters=["param1", "param2"],
        )
        assert config.name == "Test Model"
        assert config.description == "Test description"
        assert len(config.supported_parameters) == 2

    def test_config_defaults(self):
        """Test that ModelTypeConfig has correct defaults"""
        config = ModelTypeConfig(
            name="Test",
            description="Test",
            supported_parameters=[]
        )
        assert config.supports_streaming is True
        assert config.supports_tool_calling is False
        assert config.supports_vision is False
        assert len(config.required_parameters) == 0
        assert len(config.default_parameters) == 0


class TestModelTypeConfigs:
    """Test MODEL_TYPE_CONFIGS dictionary"""

    def test_all_model_types_configured(self):
        """Test that all model types have configuration"""
        model_types = [
            "CODE_GENERATOR",
            "CODE_EXPLAINER",
            "CODE_TRANSLATOR",
            "CODE_REVIEWER",
            "IMAGE_GENERATOR",
            "TEXT_COMPLETION",
            "CHAT",
            "DOCUMENT_PROCESSOR",
        ]

        for model_type in model_types:
            assert model_type in MODEL_TYPE_CONFIGS

    def test_code_generator_config(self):
        """Test Code Generator configuration"""
        config = MODEL_TYPE_CONFIGS["CODE_GENERATOR"]
        assert config.name == "Code Generator"
        assert "temperature" in config.supported_parameters
        assert "language" in config.required_parameters
        assert config.supports_streaming is True
        assert config.max_tokens == 4096

    def test_image_generator_config(self):
        """Test Image Generator configuration"""
        config = MODEL_TYPE_CONFIGS["IMAGE_GENERATOR"]
        assert config.name == "Image Generator"
        assert config.output_format == "image"
        assert config.supports_streaming is False
        assert "width" in config.required_parameters
        assert "height" in config.required_parameters

    def test_chat_config(self):
        """Test Chat configuration"""
        config = MODEL_TYPE_CONFIGS["CHAT"]
        assert config.supports_tool_calling is True
        assert config.supports_streaming is True
        assert "temperature" in config.supported_parameters

    def test_config_has_defaults(self):
        """Test that all configs have default parameters"""
        for model_type, config in MODEL_TYPE_CONFIGS.items():
            assert len(config.default_parameters) > 0, f"{model_type} has no defaults"


class TestStreamingSupport:
    """Test streaming support checking"""

    def test_streaming_supported_models(self):
        """Test models that support streaming"""
        supported = [
            "CODE_GENERATOR",
            "CODE_EXPLAINER",
            "CODE_TRANSLATOR",
            "CODE_REVIEWER",
            "TEXT_COMPLETION",
            "CHAT",
            "DOCUMENT_PROCESSOR",
        ]

        for model_type in supported:
            assert is_streaming_supported(model_type) is True

    def test_streaming_unsupported_models(self):
        """Test models that don't support streaming"""
        unsupported = ["IMAGE_GENERATOR"]

        for model_type in unsupported:
            assert is_streaming_supported(model_type) is False

    def test_streaming_invalid_model(self):
        """Test streaming check for invalid model"""
        assert is_streaming_supported("INVALID") is False


class TestToolCallingSupport:
    """Test tool calling support checking"""

    def test_tool_calling_supported(self):
        """Test models that support tool calling"""
        assert is_tool_calling_supported("CHAT") is True

    def test_tool_calling_unsupported(self):
        """Test models that don't support tool calling"""
        unsupported = [
            "CODE_GENERATOR",
            "CODE_EXPLAINER",
            "CODE_TRANSLATOR",
            "CODE_REVIEWER",
            "IMAGE_GENERATOR",
            "TEXT_COMPLETION",
            "DOCUMENT_PROCESSOR",
        ]

        for model_type in unsupported:
            assert is_tool_calling_supported(model_type) is False


class TestParameterSupport:
    """Test parameter support functions"""

    def test_get_supported_parameters(self):
        """Test get_supported_parameters function"""
        params = get_supported_parameters("CODE_GENERATOR")
        assert isinstance(params, list)
        assert len(params) > 0
        assert "temperature" in params
        assert "max_tokens" in params

    def test_get_supported_parameters_invalid(self):
        """Test get_supported_parameters for invalid type"""
        params = get_supported_parameters("INVALID")
        assert params == []

    def test_get_default_parameters(self):
        """Test get_default_parameters function"""
        defaults = get_default_parameters("CODE_GENERATOR")
        assert isinstance(defaults, dict)
        assert len(defaults) > 0
        assert "temperature" in defaults or "language" in defaults

    def test_get_default_parameters_invalid(self):
        """Test get_default_parameters for invalid type"""
        defaults = get_default_parameters("INVALID")
        assert defaults == {}

    def test_language_required_for_code_generator(self):
        """Test that CODE_GENERATOR requires language"""
        config = get_model_type_config("CODE_GENERATOR")
        assert "language" in config.required_parameters


class TestTokenLimits:
    """Test token limit functions"""

    def test_get_max_tokens(self):
        """Test get_max_tokens function"""
        # Code generator should have high limit
        tokens = get_max_tokens("CODE_GENERATOR")
        assert tokens == 4096

        # Image generator should have None
        tokens = get_max_tokens("IMAGE_GENERATOR")
        assert tokens is None

    def test_get_max_tokens_invalid(self):
        """Test get_max_tokens for invalid type"""
        tokens = get_max_tokens("INVALID")
        assert tokens is None

    def test_get_context_window(self):
        """Test get_context_window function"""
        # Most models should have context window
        window = get_context_window("CODE_GENERATOR")
        assert window == 8192

        # Image generator should have None
        window = get_context_window("IMAGE_GENERATOR")
        assert window is None

    def test_get_context_window_invalid(self):
        """Test get_context_window for invalid type"""
        window = get_context_window("INVALID")
        assert window is None

    def test_context_window_consistency(self):
        """Test that context window >= max_tokens"""
        for model_type in ["CODE_GENERATOR", "CODE_EXPLAINER", "TEXT_COMPLETION"]:
            max_tokens = get_max_tokens(model_type)
            context_window = get_context_window(model_type)

            if max_tokens and context_window:
                assert context_window >= max_tokens, \
                    f"{model_type}: context_window should be >= max_tokens"


class TestConfigConsistency:
    """Test consistency across configurations"""

    def test_all_configs_have_descriptions(self):
        """Test that all configs have descriptions"""
        for model_type, config in MODEL_TYPE_CONFIGS.items():
            assert len(config.description) > 0

    def test_all_configs_have_output_format(self):
        """Test that all configs have output format"""
        for model_type, config in MODEL_TYPE_CONFIGS.items():
            assert len(config.output_format) > 0

    def test_required_params_subset_of_supported(self):
        """Test that required params are subset of supported params"""
        for model_type, config in MODEL_TYPE_CONFIGS.items():
            for required in config.required_parameters:
                assert required in config.supported_parameters, \
                    f"{model_type}: {required} is required but not supported"

    def test_defaults_in_supported_parameters(self):
        """Test that default parameters are in supported parameters"""
        for model_type, config in MODEL_TYPE_CONFIGS.items():
            for param_name in config.default_parameters:
                assert param_name in config.supported_parameters, \
                    f"{model_type}: {param_name} has default but not supported"
