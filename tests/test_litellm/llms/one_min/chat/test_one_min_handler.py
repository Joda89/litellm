"""
Tests for 1min.ai chat completion handler
"""
from unittest.mock import MagicMock, patch
import pytest

from litellm.llms.one_min.chat.handler import OneMinChatCompletion
from litellm.llms.one_min.config import FeatureType, DEFAULT_CONFIG
from litellm.llms.one_min.exceptions import (
    OneMinAIAuthError,
    OneMinAIValidationError,
)
from litellm.types.utils import ModelResponse


class TestOneMinChatCompletionInit:
    """Test handler initialization"""

    def test_handler_initialization(self):
        handler = OneMinChatCompletion()
        assert handler is not None
        assert handler.api_base == DEFAULT_CONFIG["api_base"]

    def test_handler_has_completion_method(self):
        handler = OneMinChatCompletion()
        assert hasattr(handler, "completion")
        assert callable(handler.completion)

    def test_handler_has_acompletion_method(self):
        handler = OneMinChatCompletion()
        assert hasattr(handler, "acompletion")
        assert callable(handler.acompletion)


class TestExtractPrompt:
    """Test _extract_prompt method"""

    def setup_method(self):
        self.handler = OneMinChatCompletion()

    def test_extract_from_user_message(self):
        messages = [{"role": "user", "content": "Hello"}]
        assert self.handler._extract_prompt(messages) == "Hello"

    def test_extract_from_last_user_message(self):
        messages = [
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Response"},
            {"role": "user", "content": "Second"},
        ]
        assert self.handler._extract_prompt(messages) == "Second"

    def test_extract_from_multipart_content(self):
        messages = [{"role": "user", "content": [
            {"type": "text", "text": "Hello"},
            {"type": "text", "text": "World"},
        ]}]
        assert self.handler._extract_prompt(messages) == "Hello World"

    def test_extract_from_empty_messages(self):
        assert self.handler._extract_prompt([]) == ""

    def test_extract_fallback_to_last_message(self):
        messages = [{"role": "system", "content": "System prompt"}]
        assert self.handler._extract_prompt(messages) == "System prompt"


class TestBuildPromptObject:
    """Test _build_prompt_object method"""

    def setup_method(self):
        self.handler = OneMinChatCompletion()

    def test_chat_feature_basic(self):
        result = self.handler._build_prompt_object(
            feature_type=FeatureType.CHAT_WITH_AI.value,
            prompt="Hello",
            web_search=False,
            optional_params={}
        )
        assert result["prompt"] == "Hello"
        assert result["isMixed"] is False
        assert result["webSearch"] is False

    def test_chat_feature_with_web_search(self):
        result = self.handler._build_prompt_object(
            feature_type=FeatureType.CHAT_WITH_AI.value,
            prompt="Search this",
            web_search=True,
            optional_params={}
        )
        assert result["webSearch"] is True
        assert "numOfSite" in result
        assert "maxWord" in result

    def test_image_generator_feature(self):
        result = self.handler._build_prompt_object(
            feature_type=FeatureType.IMAGE_GENERATOR.value,
            prompt="A cat",
            web_search=False,
            optional_params={}
        )
        assert result["prompt"] == "A cat"
        assert "num_outputs" in result
        assert "aspect_ratio" in result

    def test_image_variator_feature(self):
        result = self.handler._build_prompt_object(
            feature_type=FeatureType.IMAGE_VARIATOR.value,
            prompt="http://example.com/image.png",
            web_search=False,
            optional_params={}
        )
        assert result["imageUrl"] == "http://example.com/image.png"
        assert "mode" in result

    def test_code_generator_feature(self):
        result = self.handler._build_prompt_object(
            feature_type=FeatureType.CODE_GENERATOR.value,
            prompt="Write a sort function",
            web_search=False,
            optional_params={}
        )
        assert result["prompt"] == "Write a sort function"
        assert result["webSearch"] is False
        assert "numOfSite" in result
        assert "maxWord" in result

    def test_image_to_prompt_feature(self):
        result = self.handler._build_prompt_object(
            feature_type=FeatureType.IMAGE_TO_PROMPT.value,
            prompt="http://example.com/image.png",
            web_search=False,
            optional_params={}
        )
        assert result["imageUrl"] == "http://example.com/image.png"
        assert result["mode"] == "fast"
        assert result["n"] == 1

    def test_text_to_video_feature(self):
        result = self.handler._build_prompt_object(
            feature_type=FeatureType.TEXT_TO_VIDEO.value,
            prompt="A sunset",
            web_search=False,
            optional_params={}
        )
        assert result["prompt"] == "A sunset"

    def test_content_generator_feature(self):
        result = self.handler._build_prompt_object(
            feature_type=FeatureType.CONTENT_GENERATOR.value,
            prompt="Write about AI",
            web_search=False,
            optional_params={"tone": "casual", "language": "fr"}
        )
        assert result["prompt"] == "Write about AI"
        assert result["tone"] == "casual"
        assert result["language"] == "fr"

    def test_content_generator_defaults(self):
        result = self.handler._build_prompt_object(
            feature_type=FeatureType.CONTENT_GENERATOR.value,
            prompt="Write about AI",
            web_search=False,
            optional_params={}
        )
        assert result["tone"] == "professional"
        assert result["language"] == "en"

    def test_text_to_speech_feature(self):
        result = self.handler._build_prompt_object(
            feature_type=FeatureType.TEXT_TO_SPEECH.value,
            prompt="Hello world",
            web_search=False,
            optional_params={}
        )
        assert result["prompt"] == "Hello world"

    def test_unknown_feature_returns_generic(self):
        """Unknown feature types should return a generic prompt object"""
        result = self.handler._build_prompt_object(
            feature_type="UNKNOWN_TYPE",
            prompt="Test",
            web_search=False,
            optional_params={}
        )
        assert result["prompt"] == "Test"

    def test_code_explainer_no_longer_supported(self):
        """CODE_EXPLAINER branch was removed, should fall through to generic"""
        result = self.handler._build_prompt_object(
            feature_type="CODE_EXPLAINER",
            prompt="some code",
            web_search=False,
            optional_params={}
        )
        # Should not raise, should return generic prompt object
        assert result["prompt"] == "some code"


class TestCompletionValidation:
    """Test completion method validation"""

    def setup_method(self):
        self.handler = OneMinChatCompletion()

    def test_completion_requires_api_key(self):
        with pytest.raises(OneMinAIAuthError):
            self.handler.completion(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hi"}],
                api_base="https://api.1min.ai/api/features",
                api_key="",
                model_response=MagicMock(spec=ModelResponse),
                logging_obj=MagicMock(),
                optional_params={},
            )

    def test_completion_requires_messages(self):
        with pytest.raises(OneMinAIValidationError):
            self.handler.completion(
                model="gpt-4o",
                messages=[],
                api_base="https://api.1min.ai/api/features",
                api_key="test-key",
                model_response=MagicMock(spec=ModelResponse),
                logging_obj=MagicMock(),
                optional_params={},
            )

    def test_completion_rejects_unsupported_model_for_feature(self):
        with pytest.raises(OneMinAIValidationError, match="not supported"):
            self.handler.completion(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hi"}],
                api_base="https://api.1min.ai/api/features",
                api_key="test-key",
                model_response=MagicMock(spec=ModelResponse),
                logging_obj=MagicMock(),
                optional_params={"feature_type": "IMAGE_GENERATOR"},
            )


class TestConvertResponse:
    """Test _convert_response method"""

    def setup_method(self):
        self.handler = OneMinChatCompletion()

    def test_convert_string_result(self):
        response_data = {
            "aiRecord": {
                "uuid": "test-uuid",
                "aiRecordDetail": {
                    "resultObject": "Hello response"
                }
            }
        }
        model_response = ModelResponse()
        result = self.handler._convert_response(response_data, "gpt-4o", model_response)
        assert result.choices[0].message.content == "Hello response"
        assert result.model == "gpt-4o"

    def test_convert_empty_result(self):
        response_data = {
            "aiRecord": {
                "uuid": "test-uuid",
                "aiRecordDetail": {
                    "resultObject": ""
                }
            }
        }
        model_response = ModelResponse()
        result = self.handler._convert_response(response_data, "gpt-4o", model_response)
        assert result.choices[0].message.content == ""

    def test_convert_list_result(self):
        response_data = {
            "aiRecord": {
                "uuid": "test-uuid",
                "aiRecordDetail": {
                    "resultObject": ["item1", "item2"]
                }
            }
        }
        model_response = ModelResponse()
        result = self.handler._convert_response(response_data, "gpt-4o", model_response)
        assert "item1" in result.choices[0].message.content
