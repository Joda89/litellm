"""
Custom exceptions for 1min.ai integration
"""


class OneMinAIError(Exception):
    """Base exception for 1min.ai errors"""
    pass


class OneMinAIAuthError(OneMinAIError):
    """Authentication error (invalid or missing API key)"""
    pass


class OneMinAIValidationError(OneMinAIError):
    """Validation error for input parameters"""
    pass


class OneMinAIAPIError(OneMinAIError):
    """API error from 1min.ai"""
    def __init__(self, message: str, status_code: int = None, response: dict = None):
        super().__init__(message)
        self.status_code = status_code
        self.response = response


class OneMinAIRateLimitError(OneMinAIAPIError):
    """Rate limit exceeded"""
    pass


class OneMinAITimeoutError(OneMinAIError):
    """Request timeout"""
    pass
