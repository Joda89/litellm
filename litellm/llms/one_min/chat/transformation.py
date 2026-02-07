"""
Transformation configuration for 1min.ai chat completions
"""

from litellm.llms.openai_like.chat.transformation import OpenAILikeChatConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues
from typing import List, Optional, Tuple, Union


class OneMinChatConfig(OpenAILikeChatConfig):
    """
    Configuration for 1min.ai chat completions.

    Note: 1min.ai does NOT use OpenAI-compatible format.
    This config provides parameter mapping but the actual
    transformation happens in the handler.
    """

    frequency_penalty: Optional[int] = None
    function_call: Optional[Union[str, dict]] = None
    functions: Optional[list] = None
    logit_bias: Optional[dict] = None
    max_tokens: Optional[int] = None
    n: Optional[int] = None
    presence_penalty: Optional[int] = None
    stop: Optional[Union[str, list]] = None
    temperature: Optional[int] = None
    top_p: Optional[int] = None
    response_format: Optional[dict] = None
    tools: Optional[list] = None
    tool_choice: Optional[Union[str, dict]] = None

    def __init__(
        self,
        frequency_penalty: Optional[int] = None,
        function_call: Optional[Union[str, dict]] = None,
        functions: Optional[list] = None,
        logit_bias: Optional[dict] = None,
        max_tokens: Optional[int] = None,
        n: Optional[int] = None,
        presence_penalty: Optional[int] = None,
        stop: Optional[Union[str, list]] = None,
        temperature: Optional[int] = None,
        top_p: Optional[int] = None,
        response_format: Optional[dict] = None,
        tools: Optional[list] = None,
        tool_choice: Optional[Union[str, dict]] = None,
    ) -> None:
        locals_ = locals().copy()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    def _get_openai_compatible_provider_info(
            self,
            api_base: Optional[str],
            api_key: Optional[str],
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Get provider info for 1min.ai API.

        Returns:
            Tuple of (api_base, api_key)
        """
        api_base = (
                api_base
                or get_secret_str("ONE_MIN_API_BASE")
                or "https://api.1min.ai/api/features"
        )
        dynamic_api_key = api_key or get_secret_str("ONE_MIN_API_KEY")
        return api_base, dynamic_api_key

    @property
    def custom_llm_provider(self) -> Optional[str]:
        return "one_min"

    @classmethod
    def get_config(cls):
        return super().get_config()

    def _transform_messages(
        self, messages: List[AllMessageValues], model: str, **kwargs
    ) -> List[AllMessageValues]:
        """
        Transform messages for 1min.ai API.
        The actual transformation to 1min.ai format happens in the handler.
        """
        return messages

    def _should_fake_stream(self, optional_params: dict) -> bool:
        """
        Determine if streaming should be faked.
        1min.ai supports streaming natively.
        """
        return False
