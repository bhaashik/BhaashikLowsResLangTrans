"""
Universal Translation Package

A flexible, extensible translation framework supporting multiple providers,
language pairs, and custom prompting strategies.

Basic Usage:
    from universal_translate import AnthropicProvider, PromptManager, TranslationRequest, TranslationUnit

    # Load prompt configuration
    prompt_mgr = PromptManager("config/prompts/low_resource.yaml")

    # Create provider
    provider = AnthropicProvider(
        model="claude-haiku-4.5",
        prompt_manager=prompt_mgr
    )

    # Create translation request
    request = TranslationRequest(
        units=[TranslationUnit(text="नमस्ते", index=0)],
        src_lang="hi",
        tgt_lang="bho"
    )

    # Translate
    with provider:
        response = provider.translate_sync(request)

    print(response.results[0].translation)
"""

__version__ = "0.1.0"

from .core import (
    BaseTranslator,
    TranslationUnit,
    TranslationRequest,
    TranslationResult,
    TranslationResponse,
    TranslationStatus,
    CostEstimate,
    TranslationError
)

from .prompts import PromptManager

# Import providers (optional, may fail if dependencies not installed)
try:
    from .providers import AnthropicProvider
except ImportError:
    AnthropicProvider = None

__all__ = [
    # Core
    'BaseTranslator',
    'TranslationUnit',
    'TranslationRequest',
    'TranslationResult',
    'TranslationResponse',
    'TranslationStatus',
    'CostEstimate',
    'TranslationError',

    # Prompts
    'PromptManager',

    # Providers
    'AnthropicProvider',
]
