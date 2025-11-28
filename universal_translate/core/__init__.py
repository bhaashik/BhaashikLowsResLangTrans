"""Core translation framework components."""

from .base_translator import (
    BaseTranslator,
    TranslationError,
    RateLimitError,
    AuthenticationError,
    InvalidRequestError,
    ServiceUnavailableError
)
from .translation_request import (
    TranslationUnit,
    TranslationRequest,
    TranslationResult,
    TranslationResponse,
    TranslationStatus,
    CostEstimate
)

__all__ = [
    # Base classes
    'BaseTranslator',

    # Exceptions
    'TranslationError',
    'RateLimitError',
    'AuthenticationError',
    'InvalidRequestError',
    'ServiceUnavailableError',

    # Data models
    'TranslationUnit',
    'TranslationRequest',
    'TranslationResult',
    'TranslationResponse',
    'TranslationStatus',
    'CostEstimate',
]
