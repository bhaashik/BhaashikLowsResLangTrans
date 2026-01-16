"""Language embeddings for representing typological features.

This module provides language embeddings that encode linguistic characteristics
of languages, such as:
- Typological features (word order, morphology, phonology)
- Genetic relationships (language families)
- Geographic proximity
- Writing system features

These embeddings can improve translation quality by providing the model with
information about the linguistic properties of source and target languages.

Sources:
- URIEL/lang2vec: 103 typological features for 7,000+ languages
- WALS: World Atlas of Language Structures (192 features, 2,000+ languages)
- Custom embeddings: Learned from data

References:
- Littell et al. (2017): URIEL and lang2vec
- Dryer & Haspelmath (2013): WALS
- Östling & Tiedemann (2017): Continuous Multilinguality with Language Vectors
"""

from src.training.language_embeddings.config import LanguageEmbeddingsConfig
from src.training.language_embeddings.loaders import (
    AbstractLanguageEmbeddingLoader,
    URIELLoader,
    WALSLoader,
    Lang2VecLoader,
)
from src.training.language_embeddings.encoder import LanguageEmbeddingEncoder
from src.training.language_embeddings.integration import (
    LanguageAwareEncoder,
    add_language_embeddings,
)

__all__ = [
    "LanguageEmbeddingsConfig",
    "AbstractLanguageEmbeddingLoader",
    "URIELLoader",
    "WALSLoader",
    "Lang2VecLoader",
    "LanguageEmbeddingEncoder",
    "LanguageAwareEncoder",
    "add_language_embeddings",
]
