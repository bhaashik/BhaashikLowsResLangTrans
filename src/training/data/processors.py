"""Data processors for preprocessing text."""

from typing import Dict, Any, Callable
from abc import ABC, abstractmethod


class DataProcessorRegistry:
    """
    Registry for data processors.

    Allows custom preprocessing pipelines to be registered and used.
    """

    _registry: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator for registering a processor."""
        def decorator(processor_class):
            cls._registry[name] = processor_class
            return processor_class
        return decorator

    @classmethod
    def get(cls, name: str):
        """Get processor by name."""
        if name not in cls._registry:
            raise ValueError(f"Processor '{name}' not registered")
        return cls._registry[name]

    @classmethod
    def list_processors(cls) -> list:
        """List all registered processors."""
        return list(cls._registry.keys())


class AbstractDataProcessor(ABC):
    """Abstract base class for data processors."""

    @abstractmethod
    def process(self, text: str) -> str:
        """Process text."""
        pass


@DataProcessorRegistry.register("identity")
class IdentityProcessor(AbstractDataProcessor):
    """No-op processor that returns text unchanged."""

    def process(self, text: str) -> str:
        """Return text unchanged."""
        return text


@DataProcessorRegistry.register("basic")
class BasicProcessor(AbstractDataProcessor):
    """Basic preprocessing (whitespace normalization)."""

    def __init__(self, lowercase: bool = False):
        self.lowercase = lowercase

    def process(self, text: str) -> str:
        """Apply basic preprocessing."""
        # Remove extra whitespace
        text = " ".join(text.split())

        # Lowercase if requested
        if self.lowercase:
            text = text.lower()

        return text


@DataProcessorRegistry.register("indic_trans")
class IndicTransProcessor(AbstractDataProcessor):
    """IndicTrans-style preprocessing."""

    def process(self, text: str) -> str:
        """Apply IndicTrans preprocessing."""
        # Basic normalization
        text = " ".join(text.split())

        # TODO: Add IndicTrans-specific normalization
        # - Script normalization
        # - Character normalization
        # - Punctuation handling

        return text
