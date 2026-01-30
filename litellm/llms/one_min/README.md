# 1min.ai Integration for LiteLLM

Custom handler for integrating 1min.ai's AI Feature API with LiteLLM.

## Overview

1min.ai provides a unified API endpoint for accessing multiple AI providers including OpenAI, Anthropic, Google, Mistral, and more.

**Official documentation:** https://docs.1min.ai/docs/api/ai-feature-api

## API Structure

Unlike OpenAI-compatible APIs, 1min.ai uses its own API format:

**Endpoint:**
```
POST https://api.1min.ai/api/features
POST https://api.1min.ai/api/features?isStreaming=true
```

**Request format:**
```json
{
  "type": "CHAT_WITH_AI",
  "model": "gpt-4o-mini",
  "promptObject": {
    "prompt": "Your message",
    "isMixed": false,
    "webSearch": false
  }
}
```

## Feature Types

| Type | Description |
|------|-------------|
| `CHAT_WITH_AI` | Standard chat |
| `CHAT_WITH_IMAGE` | Chat with image analysis |
| `CHAT_WITH_PDF` | Chat with PDF documents |
| `CHAT_WITH_YOUTUBE_VIDEO` | Chat with YouTube videos |
| `IMAGE_GENERATOR` | Generate images |
| `IMAGE_VARIATOR` | Create image variations |

## Available Models

### OpenAI
- gpt-5.2-pro, gpt-5.2, gpt-5.1, gpt-5, gpt-5-mini
- gpt-4o, gpt-4o-mini, gpt-4.1, gpt-4-turbo
- o3, o3-pro, o4-mini

### Anthropic
- claude-opus-4-5-20251101, claude-opus-4-20250514
- claude-sonnet-4-5-20250929, claude-sonnet-4-20250514
- claude-haiku-4-5-20251001

### Google
- gemini-3-pro-preview, gemini-2.5-pro, gemini-2.5-flash

### Mistral
- mistral-large-latest, mistral-medium-latest, mistral-small-latest

### Others
- DeepSeek: deepseek-reasoner, deepseek-chat
- Perplexity: sonar-reasoning-pro, sonar-pro
- xAI: grok-4, grok-3
- Meta: meta/meta-llama-3.1-405b-instruct
- Alibaba: qwen3-max, qwen-plus
- Cohere: command-r-08-2024

## Usage with LiteLLM

```python
from litellm import completion

response = completion(
    model="one_min/gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}],
    api_key="your-1min-api-key"
)
```

## Configuration

Set your API key:
```bash
export ONE_MIN_API_KEY="your-api-key"
```

Or in Python:
```python
import os
os.environ["ONE_MIN_API_KEY"] = "your-api-key"
```

## Files

- `__init__.py` - Module exports
- `config.py` - Configuration, enums, and constants
- `exceptions.py` - Custom exceptions
- `handler.py` - Main chat handler

## Response Format

The handler converts 1min.ai responses to LiteLLM's standard ModelResponse format, maintaining compatibility with the rest of the LiteLLM ecosystem.
