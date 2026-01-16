"""Loaders for language embeddings from different sources."""

from src.training.language_embeddings.loaders.base import (
    AbstractLanguageEmbeddingLoader,
    LanguageEmbedding,
    LoaderRegistry,
)
from src.training.language_embeddings.loaders.uriel_loader import URIELLoader
from src.training.language_embeddings.loaders.lang2vec_loader import Lang2VecLoader
from src.training.language_embeddings.loaders.wals_loader import WALSLoader

__all__ = [
    "AbstractLanguageEmbeddingLoader",
    "LanguageEmbedding",
    "LoaderRegistry",
    "URIELLoader",
    "Lang2VecLoader",
    "WALSLoader",
]
