"""
lang2vec loader for learned language embeddings.

lang2vec provides learned language representations that capture linguistic
similarity. Unlike URIEL features, these are dense learned vectors.

Reference:
    Littell et al. (2017). URIEL and lang2vec: Representing languages as
    typological, geographical, and phylogenetic vectors. EACL 2017.
"""

from typing import List, Optional
import numpy as np
import logging

from src.training.language_embeddings.loaders.base import (
    AbstractLanguageEmbeddingLoader,
    LanguageEmbedding,
    LoaderRegistry,
)

logger = logging.getLogger(__name__)


@LoaderRegistry.register("lang2vec")
class Lang2VecLoader(AbstractLanguageEmbeddingLoader):
    """
    Loader for lang2vec learned embeddings.

    Provides dense learned language representations from lang2vec.
    """

    def __init__(
        self,
        feature_types: Optional[List[str]] = None,
        normalize: bool = True,
        cache: bool = True,
    ):
        """
        Initialize lang2vec loader.

        Args:
            feature_types: Not used (lang2vec uses learned embeddings)
            normalize: Whether to normalize embeddings
            cache: Whether to cache embeddings
        """
        super().__init__(feature_types, normalize, cache)
        self.lang2vec = None

    def initialize(self):
        """Initialize lang2vec loader."""
        if self._initialized:
            return

        logger.info("Initializing lang2vec loader...")

        try:
            import lang2vec.lang2vec as l2v
            self.lang2vec = l2v
            logger.info("✓ lang2vec library loaded")
        except ImportError:
            raise ImportError(
                "lang2vec library required. Install with: pip install lang2vec"
            )

        self._initialized = True

    def supports_language(self, language_code: str) -> bool:
        """Check if language is supported."""
        if not self._initialized:
            self.initialize()

        try:
            features = self.lang2vec.get_features(language_code, "learned")
            return language_code in features
        except Exception:
            return False

    def load_embedding(self, language_code: str) -> LanguageEmbedding:
        """Load lang2vec embedding for language."""
        if not self._initialized:
            self.initialize()

        # Check cache
        if self.cache and language_code in self._embedding_cache:
            return self._embedding_cache[language_code]

        # Get learned features from lang2vec
        try:
            features_dict = self.lang2vec.get_features(
                [language_code],
                "learned",
                header=False
            )

            if language_code not in features_dict:
                raise ValueError(
                    f"Language {language_code} not found in lang2vec"
                )

            embedding_vector = np.array(
                features_dict[language_code],
                dtype=np.float32
            )

        except Exception as e:
            raise ValueError(
                f"Could not load lang2vec embedding for {language_code}: {e}"
            )

        # Create embedding object
        embedding = LanguageEmbedding(
            language_code=language_code,
            embedding=embedding_vector,
            feature_names=None,  # Learned features don't have names
            source="lang2vec",
            metadata={
                "loader": "Lang2VecLoader",
                "embedding_type": "learned",
            }
        )

        # Normalize if requested
        if self.normalize:
            embedding.normalize()

        # Cache
        if self.cache:
            self._embedding_cache[language_code] = embedding

        return embedding

    def get_available_languages(self) -> List[str]:
        """Get list of available languages."""
        logger.info("lang2vec supports 7,000+ languages")
        return ["(7000+ languages via lang2vec)"]

    def get_feature_names(self) -> List[str]:
        """Get names of features."""
        # Learned embeddings don't have interpretable feature names
        return [f"learned_{i}" for i in range(512)]  # Default dimension

    def get_embedding_dimension(self) -> int:
        """Get dimension of embeddings."""
        return 512  # lang2vec learned embeddings are 512-dimensional
