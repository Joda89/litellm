"""
Transformation configuration for 1min.ai chat completions
"""

from typing import List, Optional, Union

from litellm.types.llms.openai import AllMessageValues
from litellm.llms.openai_like.chat.transformation import OpenAILikeChatConfig


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
