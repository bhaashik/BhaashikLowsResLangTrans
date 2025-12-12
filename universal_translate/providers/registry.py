#!/usr/bin/env python3
"""
Provider registry for managing multiple LLM providers.

This provides a unified interface to create and manage translation providers
across OpenAI, Anthropic, Gemini, and other LLM services.
"""

from typing import Dict, Type, Optional, List, Any
from pathlib import Path
import os

from ..core import BaseTranslator
from ..prompts import PromptManager
from .openai_provider import OpenAIProvider
from .anthropic_provider import AnthropicProvider
from .gemini_provider import GeminiProvider


class ProviderRegistry:
    """Registry for managing translation providers."""

    # Available providers
    PROVIDERS: Dict[str, Type[BaseTranslator]] = {
        'openai': OpenAIProvider,
        'anthropic': AnthropicProvider,
        'gemini': GeminiProvider
    }

    # Default models for each provider
    DEFAULT_MODELS = {
        'openai': 'gpt-4o-mini',
        'anthropic': 'claude-haiku-4.5',
        'gemini': 'gemini-1.5-flash'
    }

    # API key environment variable names
    API_KEY_ENV_VARS = {
        'openai': 'OPENAI_API_KEY',
        'anthropic': 'ANTHROPIC_API_KEY',
        'gemini': 'GOOGLE_API_KEY'
    }

    @classmethod
    def list_providers(cls) -> List[str]:
        """List all available providers."""
        return list(cls.PROVIDERS.keys())

    @classmethod
    def list_models(cls, provider: str) -> List[str]:
        """List available models for a provider."""
        if provider not in cls.PROVIDERS:
            raise ValueError(f"Unknown provider: {provider}")

        provider_class = cls.PROVIDERS[provider]
        return list(provider_class.MODELS.keys())

    @classmethod
    def get_default_model(cls, provider: str) -> str:
        """Get default model for a provider."""
        return cls.DEFAULT_MODELS.get(provider, list(cls.PROVIDERS[provider].MODELS.keys())[0])

    @classmethod
    def create_provider(
        cls,
        provider: str,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        prompt_config: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> BaseTranslator:
        """
        Create a translation provider.

        Args:
            provider: Provider name ('openai', 'anthropic', 'gemini')
            model: Model name (uses default if not specified)
            api_key: API key (uses environment variable if not specified)
            prompt_config: Path to prompt configuration YAML file
            config: Additional configuration
            **kwargs: Provider-specific arguments

        Returns:
            Initialized provider instance

        Example:
            >>> provider = ProviderRegistry.create_provider(
            ...     provider='openai',
            ...     model='gpt-4o-mini',
            ...     prompt_config='universal_translate/config/prompts/hi_to_mwr.yaml'
            ... )
        """
        if provider not in cls.PROVIDERS:
            raise ValueError(
                f"Unknown provider: {provider}. "
                f"Available: {', '.join(cls.list_providers())}"
            )

        provider_class = cls.PROVIDERS[provider]

        # Get model
        if model is None:
            model = cls.get_default_model(provider)

        # Get API key
        if api_key is None:
            env_var = cls.API_KEY_ENV_VARS.get(provider)
            api_key = os.getenv(env_var) if env_var else None

        if not api_key:
            raise ValueError(
                f"API key required for {provider}. "
                f"Set {cls.API_KEY_ENV_VARS.get(provider)} environment variable "
                f"or pass api_key parameter."
            )

        # Load prompt manager if config provided
        prompt_manager = None
        if prompt_config:
            prompt_manager = PromptManager(config_path=prompt_config)

        # Create provider instance
        return provider_class(
            model=model,
            api_key=api_key,
            prompt_manager=prompt_manager,
            config=config,
            **kwargs
        )

    @classmethod
    def get_provider_info(cls, provider: str) -> Dict[str, Any]:
        """
        Get information about a provider.

        Args:
            provider: Provider name

        Returns:
            Provider information including models, pricing, etc.
        """
        if provider not in cls.PROVIDERS:
            raise ValueError(f"Unknown provider: {provider}")

        provider_class = cls.PROVIDERS[provider]

        return {
            'name': provider,
            'models': list(provider_class.MODELS.keys()),
            'default_model': cls.get_default_model(provider),
            'api_key_env_var': cls.API_KEY_ENV_VARS.get(provider),
            'pricing': provider_class.MODELS
        }

    @classmethod
    def auto_detect_provider(cls) -> Optional[str]:
        """
        Auto-detect available provider based on environment variables.

        Returns:
            First available provider name, or None if none found
        """
        for provider, env_var in cls.API_KEY_ENV_VARS.items():
            if os.getenv(env_var):
                return provider
        return None

    @classmethod
    def check_provider_available(cls, provider: str) -> bool:
        """
        Check if a provider is available (API key set).

        Args:
            provider: Provider name

        Returns:
            True if provider is available
        """
        env_var = cls.API_KEY_ENV_VARS.get(provider)
        return bool(os.getenv(env_var)) if env_var else False

    @classmethod
    def list_available_providers(cls) -> List[str]:
        """List providers with API keys configured."""
        return [
            provider for provider in cls.list_providers()
            if cls.check_provider_available(provider)
        ]


# Convenience functions

def create_translator(
    provider: str = 'openai',
    model: Optional[str] = None,
    source_lang: str = 'hi',
    target_lang: str = 'mwr',
    prompt_config: Optional[str] = None,
    **kwargs
) -> BaseTranslator:
    """
    Convenience function to create a translator with common defaults.

    Args:
        provider: LLM provider ('openai', 'anthropic', 'gemini')
        model: Model name (uses provider default if not specified)
        source_lang: Source language code
        target_lang: Target language code
        prompt_config: Path to prompt configuration file
        **kwargs: Additional provider arguments

    Returns:
        Initialized translator instance

    Example:
        >>> translator = create_translator(
        ...     provider='openai',
        ...     model='gpt-4o-mini',
        ...     source_lang='hi',
        ...     target_lang='mwr'
        ... )
    """
    # Auto-detect prompt config if not provided
    if prompt_config is None:
        # Try to find prompt config for this language pair
        config_dir = Path('universal_translate/config/prompts')
        if config_dir.exists():
            # Try provider-specific config first
            provider_config = config_dir / f"{source_lang}_to_{target_lang}_{provider}.yaml"
            if provider_config.exists():
                prompt_config = str(provider_config)
            else:
                # Try generic config
                generic_config = config_dir / f"{source_lang}_to_{target_lang}.yaml"
                if generic_config.exists():
                    prompt_config = str(generic_config)

    return ProviderRegistry.create_provider(
        provider=provider,
        model=model,
        prompt_config=prompt_config,
        **kwargs
    )


def list_available_models(provider: str) -> List[str]:
    """List available models for a provider."""
    return ProviderRegistry.list_models(provider)


def get_provider_pricing(provider: str) -> Dict[str, Any]:
    """Get pricing information for a provider."""
    info = ProviderRegistry.get_provider_info(provider)
    return info['pricing']
