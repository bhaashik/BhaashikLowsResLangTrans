"""Feature extraction from dependency parses."""

from src.training.linguistic.features.dependency_encoder import (
    DependencyEncoder,
    DependencyFeatures,
)

__all__ = [
    "DependencyEncoder",
    "DependencyFeatures",
]
