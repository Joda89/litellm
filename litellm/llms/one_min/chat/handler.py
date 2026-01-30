"""
Chat completion handler for 1min.ai

This handler transforms LiteLLM requests to 1min.ai API format.
1min.ai uses a different API format than OpenAI.

Based on: https://docs.1min.ai/docs/api/ai-feature-api
"""

import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Union

import httpx

from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.types.utils import ModelResponse, Usage
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLogging

from ..config import DEFAULT_CONFIG, FeatureType
from ..exceptions import (
    OneMinAIAPIError,
    OneMinAIAuthError,
    OneMinAIRateLimitError,
    OneMinAITimeoutError,
    OneMinAIValidationError,
)


logger = logging.getLogger(__name__)


class OneMinChatCompletion:
    """
    Handler for 1min.ai chat completions

    This handler converts LiteLLM's OpenAI-like format to 1min.ai's
    proprietary API format.

    1min.ai API format:
    {
        "type": "CHAT_WITH_AI",
        "model": "gpt-4o-mini",
        "promptObject": {
            "prompt": "Hello",
            "isMixed": false,
            "webSearch": false
        }
    }
    """

    def __init__(self, **kwargs):
        self.api_base = DEFAULT_CONFIG["api_base"]

    def completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        api_base: str,
        api_key: str,
        model_response: ModelResponse,
        logging_obj: LiteLLMLogging,
        optional_params: Dict[str, Any],
        litellm_params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
        custom_llm_provider: str = "one_min",
        print_verbose: Optional[Callable] = None,
        encoding: Optional[Any] = None,
        acompletion: bool = False,
        custom_prompt_dict: Optional[Dict] = None,
        **kwargs
    ) -> ModelResponse:
        """
        Make a completion request to 1min.ai
        """
        # Validate inputs
        if not api_key:
            raise OneMinAIAuthError("API key is required")
        if not messages:
            raise OneMinAIValidationError("Messages cannot be empty")

        # Determine feature type
        feature_type = optional_params.pop("feature_type", FeatureType.CHAT_WITH_AI.value)

        # Validate model supports this feature type
        from ..config import is_model_supported_for_feature, get_supported_models_for_feature

        if not is_model_supported_for_feature(model, feature_type):
            supported_models = get_supported_models_for_feature(feature_type)
            raise OneMinAIValidationError(
                f"Model {model} is not supported for feature {feature_type}. "
                f"Supported models: {', '.join(supported_models[:5])}..."
            )

        # Extract parameters
        stream = optional_params.get("stream", False)
        web_search = optional_params.pop("web_search", optional_params.pop("webSearch", False))
        conversation_id = optional_params.pop("conversation_id", optional_params.pop("conversationId", None))
        metadata = optional_params.pop("metadata", None)

        # Build the prompt from messages
        prompt = self._extract_prompt(messages)

        # Build prompt object based on feature type
        prompt_object = self._build_prompt_object(
            feature_type=feature_type,
            prompt=prompt,
            web_search=web_search,
            optional_params=optional_params
        )

        # Build the request payload
        payload = {
            "type": feature_type,
            "model": model,
            "promptObject": prompt_object
        }

        if conversation_id:
            payload["conversationId"] = conversation_id
        if metadata:
            payload["metadata"] = metadata

        # Build headers
        request_headers = {
            "API-KEY": api_key,
            "Content-Type": "application/json"
        }
        if headers:
            request_headers.update(headers)

        # Build URL - use the correct 1min.ai endpoint
        # Don't use api_base from litellm as it adds /v1
        url = DEFAULT_CONFIG["api_base"]
        if stream:
            url = f"{url}?isStreaming=true"

        # Log the request
        logging_obj.pre_call(
            input=messages,
            api_key=api_key,
            additional_args={
                "complete_input_dict": payload,
                "api_base": url,
                "headers": request_headers,
            }
        )

        # Make the request
        try:
            if client is None:
                client = HTTPHandler(timeout=timeout or DEFAULT_CONFIG["timeout"])

            response = client.post(
                url=url,
                json=payload,
                headers=request_headers,
            )

            # Handle errors
            if response.status_code == 401:
                raise OneMinAIAuthError("Invalid API key")
            elif response.status_code == 429:
                raise OneMinAIRateLimitError("Rate limit exceeded", status_code=429)
            elif response.status_code >= 400:
                error_msg = f"API error: {response.status_code}"
                try:
                    error_data = response.json()
                    error_msg = error_data.get("message", error_msg)
                except:
                    pass
                raise OneMinAIAPIError(error_msg, status_code=response.status_code)

            # Parse response
            response_data = response.json()

            # Debug: print response structure
            logger.info(f"1min.ai response: {json.dumps(response_data, indent=2)}")

            # Log the response
            logging_obj.post_call(
                input=messages,
                api_key=api_key,
                original_response=response_data,
                additional_args={"complete_input_dict": payload}
            )

            # Convert to ModelResponse
            return self._convert_response(
                response_data=response_data,
                model=model,
                model_response=model_response
            )

        except httpx.TimeoutException:
            raise OneMinAITimeoutError(f"Request timed out after {timeout}s")
        except httpx.RequestError as e:
            raise OneMinAIAPIError(f"Request failed: {str(e)}")

    def _extract_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Extract the user's prompt from messages"""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content
                elif isinstance(content, list):
                    text_parts = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                        elif isinstance(part, str):
                            text_parts.append(part)
                    return " ".join(text_parts)

        if messages:
            content = messages[-1].get("content", "")
            return content if isinstance(content, str) else str(content)
        return ""

    def _build_prompt_object(
        self,
        feature_type: str,
        prompt: str,
        web_search: bool,
        optional_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build the promptObject based on feature type"""

        if FeatureType.is_chat_feature(feature_type):
            prompt_obj = {
                "prompt": prompt,
                "isMixed": optional_params.pop("is_mixed", optional_params.pop("isMixed", False)),
                "webSearch": web_search,
            }

            if web_search:
                prompt_obj["numOfSite"] = optional_params.pop(
                    "num_sites",
                    optional_params.pop("numOfSite", DEFAULT_CONFIG["web_search_num_sites"])
                )
                prompt_obj["maxWord"] = optional_params.pop(
                    "max_words",
                    optional_params.pop("maxWord", DEFAULT_CONFIG["web_search_max_words"])
                )

            image_list = optional_params.pop("image_list", optional_params.pop("imageList", None))
            if image_list:
                prompt_obj["imageList"] = image_list

            return prompt_obj

        elif feature_type == FeatureType.IMAGE_GENERATOR.value:
            return {
                "prompt": prompt,
                "num_outputs": optional_params.pop(
                    "num_outputs",
                    optional_params.pop("n", DEFAULT_CONFIG["image_num_outputs"])
                ),
                "aspect_ratio": optional_params.pop(
                    "aspect_ratio",
                    DEFAULT_CONFIG["image_aspect_ratio"]
                ),
                "output_format": optional_params.pop(
                    "output_format",
                    DEFAULT_CONFIG["image_output_format"]
                )
            }

        elif feature_type == FeatureType.IMAGE_VARIATOR.value:
            return {
                "imageUrl": optional_params.pop("image_url", optional_params.pop("imageUrl", prompt)),
                "mode": optional_params.pop("mode", "fast"),
                "n": optional_params.pop("n", 4),
                "isNiji6": optional_params.pop("is_niji6", optional_params.pop("isNiji6", False)),
                "aspect_width": optional_params.pop("aspect_width", 1),
                "aspect_height": optional_params.pop("aspect_height", 1),
                "maintainModeration": optional_params.pop(
                    "maintain_moderation",
                    optional_params.pop("maintainModeration", True)
                )
            }

        elif feature_type == FeatureType.CODE_GENERATOR.value:
            return {
                "prompt": prompt,
                "webSearch": web_search,
                "numOfSite": optional_params.pop(
                    "num_sites",
                    optional_params.pop("numOfSite", DEFAULT_CONFIG["web_search_num_sites"])
                ),
                "maxWord": optional_params.pop(
                    "max_words",
                    optional_params.pop("maxWord", DEFAULT_CONFIG["web_search_max_words"])
                ),
            }

        elif feature_type == FeatureType.IMAGE_TO_PROMPT.value:
            return {
                "imageUrl": optional_params.pop("image_url", optional_params.pop("imageUrl", prompt)),
                "mode": optional_params.pop("mode", "fast"),
                "n": optional_params.pop("n", 1),
            }

        elif feature_type == FeatureType.TEXT_TO_VIDEO.value:
            return {
                "prompt": prompt,
            }

        elif feature_type == FeatureType.CONTENT_GENERATOR.value:
            return {
                "prompt": prompt,
                "tone": optional_params.pop("tone", "professional"),
                "language": optional_params.pop("language", "en"),
            }

        elif feature_type == FeatureType.TEXT_TO_SPEECH.value:
            return {
                "prompt": prompt,
            }

        else:
            return {
                "prompt": prompt,
            }

    def _convert_response(
        self,
        response_data: Dict[str, Any],
        model: str,
        model_response: ModelResponse
    ) -> ModelResponse:
        """Convert 1min.ai response to LiteLLM ModelResponse"""

        # Handle 1min.ai response structure
        ai_record = response_data.get("aiRecord", {})
        detail = ai_record.get("aiRecordDetail", {})
        result = detail.get("resultObject", "")

        # Convert result to string content
        if isinstance(result, str):
            content = result
        elif isinstance(result, list):
            content = json.dumps(result)
        elif result:
            content = str(result)
        else:
            content = ""

        # Build the model response
        model_response.id = ai_record.get("uuid", f"chatcmpl-{int(time.time())}")
        model_response.created = int(time.time())
        model_response.model = model
        model_response.object = "chat.completion"

        # Create choice object using namedtuple-like structure
        from collections import namedtuple
        Message = namedtuple("Message", ["role", "content"])
        Choice = namedtuple("Choice", ["index", "message", "finish_reason"])

        message = Message(role="assistant", content=content)
        choice = Choice(index=0, message=message, finish_reason="stop")

        model_response.choices = [choice]
        model_response.usage = Usage(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0
        )

        return model_response

    async def acompletion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        api_base: str,
        api_key: str,
        model_response: ModelResponse,
        logging_obj: LiteLLMLogging,
        optional_params: Dict[str, Any],
        litellm_params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
        client: Optional[AsyncHTTPHandler] = None,
        **kwargs
    ) -> ModelResponse:
        """Async version of completion"""

        if not api_key:
            raise OneMinAIAuthError("API key is required")
        if not messages:
            raise OneMinAIValidationError("Messages cannot be empty")

        feature_type = optional_params.pop("feature_type", FeatureType.CHAT_WITH_AI.value)
        stream = optional_params.get("stream", False)
        web_search = optional_params.pop("web_search", optional_params.pop("webSearch", False))
        conversation_id = optional_params.pop("conversation_id", optional_params.pop("conversationId", None))
        metadata = optional_params.pop("metadata", None)

        prompt = self._extract_prompt(messages)

        prompt_object = self._build_prompt_object(
            feature_type=feature_type,
            prompt=prompt,
            web_search=web_search,
            optional_params=optional_params
        )

        payload = {
            "type": feature_type,
            "model": model,
            "promptObject": prompt_object
        }

        if conversation_id:
            payload["conversationId"] = conversation_id
        if metadata:
            payload["metadata"] = metadata

        request_headers = {
            "API-KEY": api_key,
            "Content-Type": "application/json"
        }
        if headers:
            request_headers.update(headers)

        # Build URL - use the correct 1min.ai endpoint
        # Don't use api_base from litellm as it adds /v1
        url = DEFAULT_CONFIG["api_base"]
        if stream:
            url = f"{url}?isStreaming=true"

        logging_obj.pre_call(
            input=messages,
            api_key=api_key,
            additional_args={
                "complete_input_dict": payload,
                "api_base": url,
                "headers": request_headers,
            }
        )

        try:
            if client is None:
                client = AsyncHTTPHandler(timeout=timeout or DEFAULT_CONFIG["timeout"])

            response = await client.post(
                url=url,
                json=payload,
                headers=request_headers,
            )

            if response.status_code == 401:
                raise OneMinAIAuthError("Invalid API key")
            elif response.status_code == 429:
                raise OneMinAIRateLimitError("Rate limit exceeded", status_code=429)
            elif response.status_code >= 400:
                error_msg = f"API error: {response.status_code}"
                try:
                    error_data = response.json()
                    error_msg = error_data.get("message", error_msg)
                except:
                    pass
                raise OneMinAIAPIError(error_msg, status_code=response.status_code)

            response_data = response.json()

            logging_obj.post_call(
                input=messages,
                api_key=api_key,
                original_response=response_data,
                additional_args={"complete_input_dict": payload}
            )

            return self._convert_response(
                response_data=response_data,
                model=model,
                model_response=model_response
            )

        except httpx.TimeoutException:
            raise OneMinAITimeoutError(f"Request timed out after {timeout}s")
        except httpx.RequestError as e:
            raise OneMinAIAPIError(f"Request failed: {str(e)}")
