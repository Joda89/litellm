"""
Tests for 1min.ai chat completion handler
"""
import os
import sys
from unittest.mock import MagicMock, patch, AsyncMock
import pytest

sys.path.insert(0, os.path.abspath("../../../../../.."))

from litellm.llms.one_min.chat.handler import OneMinChatCompletion
from litellm.llms.openai_like.chat.handler import OpenAILikeChatHandler
from litellm.types.utils import ModelResponse


class TestOneMinChatCompletion:
    """Test suite for OneMinChatCompletion handler"""

    def setup_method(self):
        """Setup test fixtures"""
        self.handler = OneMinChatCompletion()
        self.model = "one_min/code-generator"

    def test_handler_initialization(self):
        """Test that OneMinChatCompletion can be initialized"""
        handler = OneMinChatCompletion()
        assert handler is not None

    def test_handler_inherits_from_openai_like(self):
        """Test that OneMinChatCompletion inherits from OpenAILikeChatHandler"""
        assert issubclass(OneMinChatCompletion, OpenAILikeChatHandler)

    def test_handler_has_completion_method(self):
        """Test that handler has completion method"""
        assert hasattr(self.handler, "completion")
        assert callable(self.handler.completion)

    @patch("litellm.llms.openai_like.chat.handler.OpenAILikeChatHandler.completion")
    def test_completion_calls_parent(self, mock_parent_completion):
        """Test that completion method calls parent class"""
        # Setup mocks
        mock_model_response = MagicMock(spec=ModelResponse)
        mock_parent_completion.return_value = mock_model_response

        messages = [{"role": "user", "content": "Hello"}]
        api_base = "https://api.1min.ai/v1"
        api_key = "test-key"

        # Call completion
        self.handler.completion(
            model=self.model,
            messages=messages,
            api_base=api_base,
            custom_llm_provider="one_min",
            custom_prompt_dict={},
            model_response=mock_model_response,
            print_verbose=print,
            encoding=None,
            api_key=api_key,
            logging_obj=MagicMock(),
            optional_params={},
        )

        # Verify parent was called
        assert mock_parent_completion.called

    def test_completion_transforms_messages(self):
        """Test that completion transforms messages"""
        with patch.object(
            self.handler.__class__.__bases__[0], "completion", return_value=MagicMock()
        ) as mock_completion:
            messages = [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Write code"},
            ]

            mock_model_response = MagicMock(spec=ModelResponse)
            mock_completion.return_value = mock_model_response

            self.handler.completion(
                model=self.model,
                messages=messages,
                api_base="https://api.1min.ai/v1",
                custom_llm_provider="one_min",
                custom_prompt_dict={},
                model_response=mock_model_response,
                print_verbose=print,
                encoding=None,
                api_key="test-key",
                logging_obj=MagicMock(),
                optional_params={},
            )

            # Verify that completion was called with transformed messages
            assert mock_completion.called
            call_kwargs = mock_completion.call_args[1]
            assert "messages" in call_kwargs

    def test_completion_with_streaming_disabled(self):
        """Test completion with streaming disabled"""
        with patch.object(
            self.handler.__class__.__bases__[0], "completion", return_value=MagicMock()
        ) as mock_completion:
            optional_params = {"stream": False}
            mock_model_response = MagicMock(spec=ModelResponse)
            mock_completion.return_value = mock_model_response

            self.handler.completion(
                model=self.model,
                messages=[{"role": "user", "content": "Hello"}],
                api_base="https://api.1min.ai/v1",
                custom_llm_provider="one_min",
                custom_prompt_dict={},
                model_response=mock_model_response,
                print_verbose=print,
                encoding=None,
                api_key="test-key",
                logging_obj=MagicMock(),
                optional_params=optional_params,
            )

            # Verify completion was called with fake_stream=False
            call_kwargs = mock_completion.call_args[1]
            assert call_kwargs.get("fake_stream") is False

    def test_completion_with_streaming_enabled(self):
        """Test completion with streaming enabled"""
        with patch.object(
            self.handler.__class__.__bases__[0], "completion", return_value=MagicMock()
        ) as mock_completion:
            optional_params = {"stream": True}
            mock_model_response = MagicMock(spec=ModelResponse)
            mock_completion.return_value = mock_model_response

            self.handler.completion(
                model=self.model,
                messages=[{"role": "user", "content": "Hello"}],
                api_base="https://api.1min.ai/v1",
                custom_llm_provider="one_min",
                custom_prompt_dict={},
                model_response=mock_model_response,
                print_verbose=print,
                encoding=None,
                api_key="test-key",
                logging_obj=MagicMock(),
                optional_params=optional_params,
            )

            # Verify completion was called
            assert mock_completion.called

    def test_completion_preserves_messages_format(self):
        """Test that message format is preserved after transformation"""
        with patch.object(
            self.handler.__class__.__bases__[0], "completion", return_value=MagicMock()
        ) as mock_completion:
            messages = [
                {"role": "user", "content": "Write a Python function"}
            ]
            mock_model_response = MagicMock(spec=ModelResponse)
            mock_completion.return_value = mock_model_response

            self.handler.completion(
                model=self.model,
                messages=messages,
                api_base="https://api.1min.ai/v1",
                custom_llm_provider="one_min",
                custom_prompt_dict={},
                model_response=mock_model_response,
                print_verbose=print,
                encoding=None,
                api_key="test-key",
                logging_obj=MagicMock(),
                optional_params={},
            )

            call_kwargs = mock_completion.call_args[1]
            transformed_messages = call_kwargs.get("messages")

            # Check that messages are preserved
            assert isinstance(transformed_messages, list)
            assert len(transformed_messages) > 0
            assert "role" in transformed_messages[0]
            assert "content" in transformed_messages[0]

    def test_completion_with_various_message_roles(self):
        """Test completion with different message roles"""
        with patch.object(
            self.handler.__class__.__bases__[0], "completion", return_value=MagicMock()
        ) as mock_completion:
            messages = [
                {"role": "system", "content": "You are an AI assistant"},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "Help me write code"},
            ]
            mock_model_response = MagicMock(spec=ModelResponse)
            mock_completion.return_value = mock_model_response

            self.handler.completion(
                model=self.model,
                messages=messages,
                api_base="https://api.1min.ai/v1",
                custom_llm_provider="one_min",
                custom_prompt_dict={},
                model_response=mock_model_response,
                print_verbose=print,
                encoding=None,
                api_key="test-key",
                logging_obj=MagicMock(),
                optional_params={},
            )

            call_kwargs = mock_completion.call_args[1]
            transformed_messages = call_kwargs.get("messages")

            # Verify all messages are preserved
            assert len(transformed_messages) == 4
            roles = [msg["role"] for msg in transformed_messages]
            assert roles == ["system", "user", "assistant", "user"]

    def test_completion_with_optional_parameters(self):
        """Test completion with optional parameters"""
        with patch.object(
            self.handler.__class__.__bases__[0], "completion", return_value=MagicMock()
        ) as mock_completion:
            optional_params = {
                "temperature": 0.7,
                "max_tokens": 500,
                "top_p": 0.9,
            }
            mock_model_response = MagicMock(spec=ModelResponse)
            mock_completion.return_value = mock_model_response

            self.handler.completion(
                model=self.model,
                messages=[{"role": "user", "content": "Hello"}],
                api_base="https://api.1min.ai/v1",
                custom_llm_provider="one_min",
                custom_prompt_dict={},
                model_response=mock_model_response,
                print_verbose=print,
                encoding=None,
                api_key="test-key",
                logging_obj=MagicMock(),
                optional_params=optional_params,
            )

            call_kwargs = mock_completion.call_args[1]
            assert call_kwargs.get("optional_params") == optional_params

    def test_completion_without_api_key(self):
        """Test completion with None api_key"""
        with patch.object(
            self.handler.__class__.__bases__[0], "completion", return_value=MagicMock()
        ) as mock_completion:
            mock_model_response = MagicMock(spec=ModelResponse)
            mock_completion.return_value = mock_model_response

            self.handler.completion(
                model=self.model,
                messages=[{"role": "user", "content": "Hello"}],
                api_base="https://api.1min.ai/v1",
                custom_llm_provider="one_min",
                custom_prompt_dict={},
                model_response=mock_model_response,
                print_verbose=print,
                encoding=None,
                api_key=None,
                logging_obj=MagicMock(),
                optional_params={},
            )

            assert mock_completion.called


class TestOneMinHandlerIntegration:
    """Integration tests for handler with configuration"""

    def test_handler_with_config(self):
        """Test that handler can use configuration"""
        from litellm.llms.one_min.chat.transformation import OneMinChatConfig

        handler = OneMinChatCompletion()
        config = OneMinChatConfig(temperature=0.7)

        assert handler is not None
        assert config.custom_llm_provider == "one_min"

    def test_handler_initialization_kwargs(self):
        """Test handler initialization with kwargs"""
        handler = OneMinChatCompletion(timeout=30, max_retries=3)
        assert handler is not None
