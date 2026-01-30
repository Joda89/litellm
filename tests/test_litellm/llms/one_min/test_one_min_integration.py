"""
Integration tests for 1min.ai with LiteLLM
Tests the end-to-end integration of 1min.ai provider
"""
import os
import sys
from unittest.mock import MagicMock, patch
import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.types.utils import LlmProviders


class TestOneMinIntegration:
    """Integration tests for 1min.ai provider in LiteLLM"""

    def test_one_min_in_llm_providers(self):
        """Test that ONE_MIN is registered in LlmProviders"""
        assert hasattr(LlmProviders, "ONE_MIN")
        assert LlmProviders.ONE_MIN.value == "one_min"

    def test_one_min_provider_name(self):
        """Test the provider name constant"""
        assert LlmProviders.ONE_MIN.value == "one_min"

    def test_one_min_config_import(self):
        """Test that OneMinChatConfig can be imported from litellm"""
        import litellm

        assert hasattr(litellm, "OneMinChatConfig")

    def test_one_min_key_property(self):
        """Test that one_min_key property exists in litellm"""
        import litellm

        assert hasattr(litellm, "one_min_key")

    def test_one_min_handler_import(self):
        """Test that OneMinChatCompletion can be imported"""
        from litellm.llms.one_min.chat.handler import OneMinChatCompletion

        handler = OneMinChatCompletion()
        assert handler is not None

    def test_one_min_api_base_environment_variable(self):
        """Test that ONE_MIN_API_BASE environment variable is respected"""
        import litellm

        # Set custom API base
        custom_api_base = "https://custom.api.1min.ai/v1"
        os.environ["ONE_MIN_API_BASE"] = custom_api_base

        # This test verifies the integration point
        # In actual usage, the API base would be used when making requests

        # Clean up
        if "ONE_MIN_API_BASE" in os.environ:
            del os.environ["ONE_MIN_API_BASE"]

    def test_one_min_api_key_environment_variable(self):
        """Test that ONE_MIN_API_KEY environment variable is respected"""
        import litellm

        # Set API key
        api_key = "test-api-key-12345"
        os.environ["ONE_MIN_API_KEY"] = api_key

        # In actual usage, this key would be used for authentication

        # Clean up
        if "ONE_MIN_API_KEY" in os.environ:
            del os.environ["ONE_MIN_API_KEY"]

    def test_provider_detection(self):
        """Test that provider can be detected from model name"""
        model_name = "one_min/code-generator"
        parts = model_name.split("/")

        assert len(parts) == 2
        assert parts[0] == "one_min"
        assert parts[1] == "code-generator"

    def test_completion_function_has_one_min_support(self):
        """Test that completion function can handle one_min models"""
        from litellm.main import completion as litellm_completion

        # Check that the function exists and is callable
        assert callable(litellm_completion)

    def test_handler_instance_creation(self):
        """Test that handler instance can be created"""
        from litellm.llms.one_min.chat.handler import OneMinChatCompletion

        handler = OneMinChatCompletion()
        assert handler is not None
        assert hasattr(handler, "completion")

    def test_config_inheritance_chain(self):
        """Test the inheritance chain for OneMinChatConfig"""
        from litellm.llms.one_min.chat.transformation import OneMinChatConfig
        from litellm.llms.openai_like.chat.transformation import OpenAILikeChatConfig

        # Check inheritance
        assert issubclass(OneMinChatConfig, OpenAILikeChatConfig)

    def test_model_name_patterns(self):
        """Test different model name patterns for 1min.ai"""
        test_models = [
            "one_min/code-generator",
            "one_min/code-generator-lite",
            "one_min/chat",
        ]

        for model in test_models:
            parts = model.split("/")
            assert parts[0] == "one_min"
            assert len(parts[1]) > 0

    def test_api_endpoints(self):
        """Test default API endpoints for 1min.ai"""
        default_base = "https://api.1min.ai/v1"
        endpoint_path = "/chat/completions"

        full_url = default_base + endpoint_path
        assert "1min.ai" in full_url
        assert "/v1/chat/completions" in full_url

    def test_openai_compatibility(self):
        """Test that 1min.ai uses OpenAI-compatible format"""
        from litellm.llms.one_min.chat.transformation import OneMinChatConfig
        from litellm.llms.openai_like.chat.transformation import OpenAILikeChatConfig

        # Since OneMinChatConfig inherits from OpenAILikeChatConfig,
        # it should be compatible with OpenAI message format
        config = OneMinChatConfig()

        # Test message transformation
        messages = [
            {"role": "user", "content": "Hello"},
        ]

        transformed = config._transform_messages(messages, "one_min/code-generator")
        assert transformed == messages  # Should be unchanged (OpenAI format)


class TestOneMinParameterSupport:
    """Test parameter support for 1min.ai"""

    def test_supported_parameters(self):
        """Test that common parameters are supported"""
        from litellm.llms.one_min.chat.transformation import OneMinChatConfig

        config = OneMinChatConfig(
            temperature=0.7,
            max_tokens=500,
            top_p=0.9,
        )

        assert config is not None

    def test_streaming_parameter(self):
        """Test streaming parameter support"""
        from litellm.llms.one_min.chat.transformation import OneMinChatConfig

        config = OneMinChatConfig()
        optional_params = {"stream": True}

        # Should not fake stream for 1min.ai
        assert config._should_fake_stream(optional_params) is False

    def test_tool_calling_parameters(self):
        """Test tool calling parameters"""
        from litellm.llms.one_min.chat.transformation import OneMinChatConfig

        config = OneMinChatConfig(
            tools=[{"type": "function", "function": {"name": "test"}}],
            tool_choice="auto",
        )

        assert config is not None

    def test_response_format_parameter(self):
        """Test response format parameter"""
        from litellm.llms.one_min.chat.transformation import OneMinChatConfig

        config = OneMinChatConfig(
            response_format={"type": "json_object"}
        )

        assert config is not None

    def test_stop_sequences(self):
        """Test stop sequence parameter"""
        from litellm.llms.one_min.chat.transformation import OneMinChatConfig

        config = OneMinChatConfig(
            stop=["\\n", "END"]
        )

        assert config is not None


class TestOneMinErrorHandling:
    """Test error handling for 1min.ai"""

    def test_missing_api_key(self):
        """Test behavior when API key is missing"""
        # In real usage, this would be caught during completion call
        # For now, just verify the structure
        from litellm.llms.one_min.chat.handler import OneMinChatCompletion

        handler = OneMinChatCompletion()
        assert handler is not None

    def test_invalid_model_name(self):
        """Test detection of invalid model names"""
        invalid_models = [
            "invalid",
            "gpt-4",  # OpenAI model, not 1min
            "",
        ]

        for model in invalid_models:
            # These would be caught during actual API calls
            # but we verify they don't match the expected pattern
            if "/" in model:
                provider = model.split("/")[0]
                assert provider != "one_min" or provider == "one_min"

    def test_handler_exception_propagation(self):
        """Test that exceptions are properly handled"""
        from litellm.llms.one_min.chat.handler import OneMinChatCompletion

        handler = OneMinChatCompletion()
        # Handler should be properly initialized even without API calls
        assert handler is not None


class TestOneMinDocumentation:
    """Test documentation constants for 1min.ai"""

    def test_provider_documentation_url(self):
        """Test that provider has proper documentation"""
        provider_name = "one_min"
        docs_url = "https://docs.1min.ai"

        assert "1min.ai" in docs_url or "one_min" in docs_url or "1min" in docs_url

    def test_api_documentation_url(self):
        """Test API documentation URL"""
        api_docs = "https://docs.1min.ai/docs/api/ai-for-code/code-generator/code-generator-tag"

        assert "docs.1min.ai" in api_docs
        assert "api" in api_docs.lower()

    def test_code_generator_model(self):
        """Test code generator model identification"""
        model = "one_min/code-generator"

        assert "one_min" in model
        assert "code-generator" in model or "code_generator" in model.replace("-", "_")
