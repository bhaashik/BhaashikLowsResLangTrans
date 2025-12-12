#!/usr/bin/env python3
"""
Base provider abstraction for LLM translation services.

This defines the interface that all LLM providers must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from pathlib import Path


@dataclass
class TranslationRequest:
    """Request for translation."""
    text: str
    source_language: str
    target_language: str
    system_prompt: str
    examples: Optional[List[Dict[str, str]]] = None
    temperature: float = 0.3
    max_tokens: int = 2048
    top_p: float = 0.9


@dataclass
class TranslationResponse:
    """Response from translation."""
    translated_text: str
    provider: str
    model: str
    usage: Dict[str, int]
    cost_estimate: Optional[Dict[str, float]] = None
    cached_tokens: int = 0
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ProviderConfig:
    """Configuration for a provider."""
    api_key: str
    model: str
    supports_caching: bool = False
    cache_discount: float = 0.0  # e.g., 0.5 for 50%, 0.9 for 90%
    pricing: Optional[Dict[str, float]] = None  # per 1M tokens in INR
    extra_params: Optional[Dict[str, Any]] = None


class BaseProvider(ABC):
    """Base class for all LLM providers."""

    def __init__(self, config: ProviderConfig):
        """
        Initialize the provider.

        Args:
            config: Provider configuration
        """
        self.config = config
        self.provider_name = self.__class__.__name__.replace('Provider', '').lower()

    @abstractmethod
    def translate(self, request: TranslationRequest) -> TranslationResponse:
        """
        Translate text using the provider's API.

        Args:
            request: Translation request

        Returns:
            Translation response
        """
        pass

    @abstractmethod
    def validate_config(self) -> bool:
        """
        Validate that the provider is properly configured.

        Returns:
            True if configuration is valid
        """
        pass

    def supports_caching(self) -> bool:
        """Check if provider supports prompt caching."""
        return self.config.supports_caching

    def format_system_prompt(self, system_prompt: str, examples: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Format system prompt with examples.

        Args:
            system_prompt: Base system prompt
            examples: Optional list of examples

        Returns:
            Formatted system prompt
        """
        if not examples:
            return system_prompt

        examples_text = "\n\nEXAMPLES:\n"
        for i, example in enumerate(examples, 1):
            source = example.get('source', example.get('text', ''))
            target = example.get('target', example.get('translation', ''))

            if source and target:
                examples_text += f"\n{i}. Source: {source}\n   Target: {target}\n"
            elif source:
                # Monolingual example
                examples_text += f"\n{i}. {source}\n"

        return f"{system_prompt}{examples_text}"

    def calculate_cost(self, usage: Dict[str, int], cached_tokens: int = 0) -> Dict[str, float]:
        """
        Calculate cost estimate in INR.

        Args:
            usage: Token usage dictionary
            cached_tokens: Number of cached tokens

        Returns:
            Cost breakdown dictionary
        """
        if not self.config.pricing:
            return {}

        pricing = self.config.pricing
        prompt_tokens = usage.get('prompt_tokens', 0)
        completion_tokens = usage.get('completion_tokens', 0)

        uncached_tokens = prompt_tokens - cached_tokens

        # Calculate costs per million tokens
        uncached_cost = uncached_tokens * pricing.get('input', 0) / 1_000_000
        cached_cost = cached_tokens * pricing.get('cached_input', pricing.get('input', 0)) / 1_000_000
        output_cost = completion_tokens * pricing.get('output', 0) / 1_000_000

        total_cost = uncached_cost + cached_cost + output_cost

        # Calculate savings
        if cached_tokens > 0 and self.config.supports_caching:
            full_price = cached_tokens * pricing.get('input', 0) / 1_000_000
            savings = full_price - cached_cost
        else:
            savings = 0

        return {
            'uncached_input_cost': uncached_cost,
            'cached_input_cost': cached_cost,
            'output_cost': output_cost,
            'total_cost': total_cost,
            'savings': savings,
            'currency': 'INR'
        }

    @classmethod
    def get_provider_info(cls) -> Dict[str, Any]:
        """
        Get information about this provider.

        Returns:
            Provider information dictionary
        """
        return {
            'name': cls.__name__.replace('Provider', ''),
            'supports_caching': False,
            'models': [],
            'pricing': {}
        }
