"""
Pytest configuration for 1min.ai tests
"""
import pytest
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))


@pytest.fixture
def one_min_config():
    """Fixture pour OneMinChatConfig"""
    from litellm.llms.one_min.chat.transformation import OneMinChatConfig
    return OneMinChatConfig()


@pytest.fixture
def one_min_handler():
    """Fixture pour OneMinChatCompletion handler"""
    from litellm.llms.one_min.chat.handler import OneMinChatCompletion
    return OneMinChatCompletion()


@pytest.fixture
def sample_messages():
    """Fixture pour des messages d'exemple"""
    return [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Write a Python function"},
        {"role": "assistant", "content": "Here's a Python function..."},
    ]


@pytest.fixture
def mock_logging_obj():
    """Fixture pour un objet logging mocké"""
    from unittest.mock import MagicMock
    return MagicMock()


@pytest.fixture
def mock_model_response():
    """Fixture pour une réponse de modèle mockée"""
    from unittest.mock import MagicMock
    return MagicMock()
