"""Base class for language embedding loaders."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class LanguageEmbedding:
    """
    Language embedding with metadata.

    Attributes:
        language_code: ISO 639-3 language code
        embedding: Feature vector (numpy array)
        feature_names: Names of features (optional)
        source: Source of embedding (uriel, wals, etc.)
        metadata: Additional metadata
    """

    language_code: str
    embedding: np.ndarray
    feature_names: Optional[List[str]] = None
    source: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate embedding."""
        if not isinstance(self.embedding, np.ndarray):
            self.embedding = np.array(self.embedding)

        if self.embedding.ndim != 1:
            raise ValueError(
                f"Embedding must be 1-dimensional, got shape {self.embedding.shape}"
            )

        if self.feature_names and len(self.feature_names) != len(self.embedding):
            raise ValueError(
                f"Number of feature names ({len(self.feature_names)}) "
                f"must match embedding dimension ({len(self.embedding)})"
            )

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return len(self.embedding)

    def normalize(self):
        """Normalize embedding to unit length."""
        norm = np.linalg.norm(self.embedding)
        if norm > 0:
            self.embedding = self.embedding / norm

    def select_features(self, indices: List[int]) -> "LanguageEmbedding":
        """
        Select subset of features.

        Args:
            indices: Indices of features to select

        Returns:
            New LanguageEmbedding with selected features
        """
        selected_embedding = self.embedding[indices]
        selected_names = None
        if self.feature_names:
            selected_names = [self.feature_names[i] for i in indices]

        return LanguageEmbedding(
            language_code=self.language_code,
            embedding=selected_embedding,
            feature_names=selected_names,
            source=self.source,
            metadata=self.metadata.copy(),
        )


class AbstractLanguageEmbeddingLoader(ABC):
    """
    Abstract base class for language embedding loaders.

    Loaders are responsible for:
    1. Loading language embeddings from various sources
    2. Mapping language codes to embeddings
    3. Providing feature metadata
    """

    def __init__(
        self,
        feature_types: Optional[List[str]] = None,
        normalize: bool = True,
        cache: bool = True,
    ):
        """
        Initialize loader.

        Args:
            feature_types: Types of features to load (source-specific)
            normalize: Whether to normalize embeddings
            cache: Whether to cache loaded embeddings
        """
        self.feature_types = feature_types or []
        self.normalize = normalize
        self.cache = cache
        self._embedding_cache: Dict[str, LanguageEmbedding] = {}
        self._initialized = False

    @abstractmethod
    def initialize(self):
        """Initialize loader (download data, load files, etc.)."""
        pass

    @abstractmethod
    def supports_language(self, language_code: str) -> bool:
        """
        Check if language is supported.

        Args:
            language_code: ISO 639-3 language code

        Returns:
            True if language is supported
        """
        pass

    @abstractmethod
    def load_embedding(self, language_code: str) -> LanguageEmbedding:
        """
        Load embedding for a language.

        Args:
            language_code: ISO 639-3 language code

        Returns:
            LanguageEmbedding object

        Raises:
            ValueError: If language is not supported
        """
        pass

    def load_embeddings(
        self,
        language_codes: List[str]
    ) -> Dict[str, LanguageEmbedding]:
        """
        Load embeddings for multiple languages.

        Args:
            language_codes: List of ISO 639-3 language codes

        Returns:
            Dictionary mapping language codes to embeddings
        """
        embeddings = {}
        for code in language_codes:
            try:
                embeddings[code] = self.load_embedding(code)
            except ValueError as e:
                logger.warning(f"Could not load embedding for {code}: {e}")

        return embeddings

    @abstractmethod
    def get_available_languages(self) -> List[str]:
        """
        Get list of available language codes.

        Returns:
            List of ISO 639-3 language codes
        """
        pass

    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """
        Get names of features in embeddings.

        Returns:
            List of feature names
        """
        pass

    def get_embedding_dimension(self) -> int:
        """
        Get dimension of embeddings.

        Returns:
            Embedding dimension
        """
        feature_names = self.get_feature_names()
        return len(feature_names)

    def cleanup(self):
        """Cleanup resources."""
        self._embedding_cache.clear()


class LoaderRegistry:
    """Registry for language embedding loaders."""

    _loaders: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str):
        """
        Decorator to register a loader.

        Args:
            name: Loader name (e.g., "uriel", "wals")
        """
        def decorator(loader_class):
            cls._loaders[name] = loader_class
            return loader_class
        return decorator

    @classmethod
    def create_loader(
        cls,
        name: str,
        **kwargs
    ) -> AbstractLanguageEmbeddingLoader:
        """
        Create a loader by name.

        Args:
            name: Loader name
            **kwargs: Arguments for loader constructor

        Returns:
            Loader instance
        """
        if name not in cls._loaders:
            raise ValueError(
                f"Unknown loader: {name}. "
                f"Available: {list(cls._loaders.keys())}"
            )

        loader_class = cls._loaders[name]
        loader = loader_class(**kwargs)
        loader.initialize()
        return loader

    @classmethod
    def get_available_loaders(cls) -> List[str]:
        """Get list of available loader names."""
        return list(cls._loaders.keys())
