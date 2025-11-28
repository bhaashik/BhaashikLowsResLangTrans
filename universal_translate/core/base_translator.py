"""Base translator interface that all providers must implement."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from .translation_request import (
    TranslationRequest,
    TranslationResponse,
    CostEstimate
)


class BaseTranslator(ABC):
    """
    Abstract base class for all translation providers.

    All translation providers (LLM APIs, translation services, local models)
    must implement this interface to be compatible with the translation pipeline.
    """

    def __init__(
        self,
        name: str,
        model: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize translator.

        Args:
            name: Provider name (e.g., 'anthropic', 'google', 'indictrans2')
            model: Model name if applicable (e.g., 'claude-haiku-4.5')
            config: Provider-specific configuration
        """
        self.name = name
        self.model = model
        self.config = config or {}
        self._initialized = False

    @abstractmethod
    async def translate(self, request: TranslationRequest) -> TranslationResponse:
        """
        Translate text units.

        Args:
            request: Translation request with units, languages, prompts

        Returns:
            Translation response with results

        Raises:
            TranslationError: If translation fails
        """
        pass

    @abstractmethod
    def translate_sync(self, request: TranslationRequest) -> TranslationResponse:
        """
        Synchronous translation (for non-async providers).

        Args:
            request: Translation request

        Returns:
            Translation response
        """
        pass

    @abstractmethod
    def get_cost_estimate(self, request: TranslationRequest) -> CostEstimate:
        """
        Estimate cost before translation.

        Args:
            request: Translation request

        Returns:
            Cost estimate with breakdown
        """
        pass

    @abstractmethod
    def supports_batch(self) -> bool:
        """
        Whether this provider supports batch translation.

        Returns:
            True if batch translation is supported
        """
        pass

    @abstractmethod
    def supports_prompt_caching(self) -> bool:
        """
        Whether this provider supports prompt caching.

        Returns:
            True if prompt caching is supported
        """
        pass

    def supports_streaming(self) -> bool:
        """
        Whether this provider supports streaming responses.

        Returns:
            True if streaming is supported
        """
        return False

    @abstractmethod
    def get_max_tokens(self) -> int:
        """
        Get maximum tokens for this provider/model.

        Returns:
            Maximum token limit
        """
        pass

    @abstractmethod
    def get_rate_limits(self) -> Dict[str, int]:
        """
        Get rate limits for this provider.

        Returns:
            Dictionary with rate limit information:
                - requests_per_minute
                - tokens_per_minute
                - etc.
        """
        pass

    def initialize(self):
        """
        Initialize provider (load models, setup connections, etc.).

        Override this method for providers that need initialization.
        """
        self._initialized = True

    def cleanup(self):
        """
        Cleanup resources (close connections, unload models, etc.).

        Override this method for providers that need cleanup.
        """
        self._initialized = False

    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()

    def __repr__(self) -> str:
        """String representation."""
        model_str = f" ({self.model})" if self.model else ""
        return f"{self.__class__.__name__}('{self.name}'{model_str})"


class TranslationError(Exception):
    """Base exception for translation errors."""

    def __init__(self, message: str, provider: str = "", details: Optional[Dict] = None):
        """
        Initialize translation error.

        Args:
            message: Error message
            provider: Provider name where error occurred
            details: Additional error details
        """
        super().__init__(message)
        self.provider = provider
        self.details = details or {}


class RateLimitError(TranslationError):
    """Rate limit exceeded."""
    pass


class AuthenticationError(TranslationError):
    """Authentication failed."""
    pass


class InvalidRequestError(TranslationError):
    """Invalid request parameters."""
    pass


class ServiceUnavailableError(TranslationError):
    """Service temporarily unavailable."""
    pass
