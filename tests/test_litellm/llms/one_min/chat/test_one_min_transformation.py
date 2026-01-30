"""
Tests for 1min.ai transformation module
"""
import os
import sys
from unittest.mock import MagicMock, patch
import pytest

sys.path.insert(0, os.path.abspath("../../../../../.."))

from litellm.llms.one_min.chat.transformation import OneMinChatConfig


class TestOneMinChatConfig:
    """Test suite for OneMinChatConfig"""

    def setup_method(self):
        """Setup test fixtures"""
        self.config = OneMinChatConfig()
        self.model = "one_min/code-generator"
        self.logging_obj = MagicMock()

    def test_config_initialization(self):
        """Test that OneMinChatConfig can be initialized with default parameters"""
        config = OneMinChatConfig()
        assert config is not None
        assert config.custom_llm_provider == "one_min"

    def test_custom_llm_provider(self):
        """Test that custom_llm_provider returns 'one_min'"""
        assert self.config.custom_llm_provider == "one_min"

    def test_initialization_with_parameters(self):
        """Test initialization with various parameters"""
        config = OneMinChatConfig(
            temperature=0.7,
            max_tokens=500,
            top_p=0.9,
            frequency_penalty=0.5,
            presence_penalty=0.5,
        )
        assert config.custom_llm_provider == "one_min"

    def test_transform_messages_basic(self):
        """Test that _transform_messages returns messages unchanged (OpenAI format)"""
        messages = [
            {"role": "user", "content": "Write a Python function"},
            {"role": "assistant", "content": "Here's a Python function..."},
        ]

        result = self.config._transform_messages(
            messages=messages, model=self.model
        )

        assert result == messages

    def test_transform_messages_with_system(self):
        """Test transform_messages with system message"""
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Write a function"},
        ]

        result = self.config._transform_messages(
            messages=messages, model=self.model
        )

        assert result == messages
        assert len(result) == 2
        assert result[0]["role"] == "system"

    def test_transform_messages_empty(self):
        """Test transform_messages with empty list"""
        messages = []
        result = self.config._transform_messages(
            messages=messages, model=self.model
        )
        assert result == []

    def test_should_fake_stream_false(self):
        """Test that _should_fake_stream returns False (1min.ai supports streaming)"""
        optional_params = {"stream": True}
        result = self.config._should_fake_stream(optional_params)
        assert result is False

    def test_should_fake_stream_with_different_params(self):
        """Test should_fake_stream with various optional params"""
        test_cases = [
            ({"stream": True}, False),
            ({"stream": False}, False),
            ({}, False),
            ({"stream": True, "temperature": 0.7}, False),
        ]

        for optional_params, expected in test_cases:
            result = self.config._should_fake_stream(optional_params)
            assert result == expected, f"Failed for params: {optional_params}"

    def test_get_config_returns_config_dict(self):
        """Test that get_config returns a dictionary"""
        config_dict = OneMinChatConfig.get_config()
        assert isinstance(config_dict, dict)

    def test_config_with_tool_calling(self):
        """Test config with tool calling parameters"""
        config = OneMinChatConfig(
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather information",
                        "parameters": {"type": "object"},
                    },
                }
            ],
            tool_choice="auto",
        )
        assert config.custom_llm_provider == "one_min"

    def test_config_with_response_format(self):
        """Test config with JSON response format"""
        config = OneMinChatConfig(
            response_format={"type": "json_object"}
        )
        assert config.custom_llm_provider == "one_min"

    def test_multiple_initialization_same_class(self):
        """Test that multiple instances don't interfere with each other"""
        config1 = OneMinChatConfig(temperature=0.5)
        config2 = OneMinChatConfig(temperature=0.9)

        # Both should be valid instances
        assert config1.custom_llm_provider == "one_min"
        assert config2.custom_llm_provider == "one_min"

    def test_transform_messages_preserves_order(self):
        """Test that message order is preserved"""
        messages = [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
            {"role": "user", "content": "Second question"},
            {"role": "assistant", "content": "Second answer"},
        ]

        result = self.config._transform_messages(
            messages=messages, model=self.model
        )

        assert len(result) == 4
        assert result[0]["content"] == "First question"
        assert result[3]["content"] == "Second answer"

    def test_config_parameter_types(self):
        """Test various parameter types"""
        config = OneMinChatConfig(
            temperature=0.7,  # float/int
            max_tokens=500,  # int
            stop=["\\n", "END"],  # list
            frequency_penalty=0.5,  # float
            presence_penalty=0.5,  # float
        )
        assert config.custom_llm_provider == "one_min"

    def test_config_with_none_parameters(self):
        """Test config initialization with None parameters"""
        config = OneMinChatConfig(
            temperature=None,
            max_tokens=None,
            top_p=None,
        )
        assert config is not None
        assert config.custom_llm_provider == "one_min"


class TestOneMinChatTransformationIntegration:
    """Integration tests for transformation with OpenAI-like base"""

    def setup_method(self):
        """Setup test fixtures"""
        self.config = OneMinChatConfig()
        self.model = "one_min/code-generator"

    def test_config_inherits_from_openai_like(self):
        """Test that OneMinChatConfig properly inherits from OpenAILikeChatConfig"""
        from litellm.llms.openai_like.chat.transformation import OpenAILikeChatConfig

        assert issubclass(OneMinChatConfig, OpenAILikeChatConfig)

    def test_has_openai_methods(self):
        """Test that inherited methods are available"""
        config = OneMinChatConfig()

        # Check for inherited methods
        assert hasattr(config, "map_openai_params")
        assert hasattr(config, "_transform_messages")
        assert hasattr(config, "_should_fake_stream")

    def test_openai_compatibility(self):
        """Test that 1min.ai messages are compatible with OpenAI format"""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}},
                ],
            }
        ]

        result = self.config._transform_messages(
            messages=messages, model=self.model
        )

        # Should preserve the complex message structure
        assert result[0]["role"] == "user"
        assert isinstance(result[0]["content"], list)
        assert len(result[0]["content"]) == 2
