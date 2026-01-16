"""Configuration for language embeddings."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import yaml
from pathlib import Path


@dataclass
class LanguageEmbeddingsConfig:
    """
    Configuration for language embeddings in MT.

    Attributes:
        use_source_embedding: Whether to use source language embedding
        use_target_embedding: Whether to use target language embedding
        use_both: Whether to use both source and target embeddings
        embedding_source: Source of embeddings ("uriel", "wals", "lang2vec", "custom")
        embedding_dim: Dimension to project embeddings to
        feature_types: Which feature types to use (if applicable)
        integration_method: How to integrate embeddings
        normalize_embeddings: Whether to normalize embeddings
        cache_embeddings: Whether to cache loaded embeddings
        custom_embedding_file: Path to custom embeddings (if using custom)
        use_feature_selection: Whether to select most relevant features
        num_selected_features: Number of features to select (if selection enabled)
    """

    # Enable/disable embeddings
    use_source_embedding: bool = False
    use_target_embedding: bool = False
    use_both: bool = False

    # Embedding source
    embedding_source: str = "uriel"  # uriel, wals, lang2vec, custom
    embedding_dim: int = 64  # Projection dimension

    # Feature selection
    feature_types: List[str] = field(default_factory=lambda: [
        "syntax",      # Syntactic features (word order, etc.)
        "phonology",   # Phonological features
        "morphology",  # Morphological features
    ])

    # Integration method
    integration_method: str = "concatenate"
    # Options:
    # - concatenate: Concatenate with encoder output
    # - add: Add to encoder output (requires same dimension)
    # - condition: Use as conditioning vector (via cross-attention)
    # - adapter: Use adapter layers conditioned on language embedding

    # Feature selection
    use_feature_selection: bool = False
    num_selected_features: int = 20

    # Normalization and caching
    normalize_embeddings: bool = True
    cache_embeddings: bool = True

    # Custom embeddings
    custom_embedding_file: Optional[str] = None

    # Custom parameters
    custom_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration."""
        if self.use_both:
            self.use_source_embedding = True
            self.use_target_embedding = True

        if not self.use_source_embedding and not self.use_target_embedding:
            raise ValueError(
                "Must enable at least one of: use_source_embedding, use_target_embedding"
            )

        valid_sources = ["uriel", "wals", "lang2vec", "custom"]
        if self.embedding_source not in valid_sources:
            raise ValueError(
                f"Invalid embedding_source: {self.embedding_source}. "
                f"Must be one of: {valid_sources}"
            )

        if self.embedding_source == "custom" and not self.custom_embedding_file:
            raise ValueError(
                "custom_embedding_file must be provided when using custom embeddings"
            )

        valid_methods = ["concatenate", "add", "condition", "adapter"]
        if self.integration_method not in valid_methods:
            raise ValueError(
                f"Invalid integration_method: {self.integration_method}. "
                f"Must be one of: {valid_methods}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "use_source_embedding": self.use_source_embedding,
            "use_target_embedding": self.use_target_embedding,
            "use_both": self.use_both,
            "embedding_source": self.embedding_source,
            "embedding_dim": self.embedding_dim,
            "feature_types": self.feature_types,
            "integration_method": self.integration_method,
            "use_feature_selection": self.use_feature_selection,
            "num_selected_features": self.num_selected_features,
            "normalize_embeddings": self.normalize_embeddings,
            "cache_embeddings": self.cache_embeddings,
            "custom_embedding_file": self.custom_embedding_file,
            **self.custom_params,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "LanguageEmbeddingsConfig":
        """Create config from dictionary."""
        known_fields = {
            "use_source_embedding", "use_target_embedding", "use_both",
            "embedding_source", "embedding_dim", "feature_types",
            "integration_method", "use_feature_selection", "num_selected_features",
            "normalize_embeddings", "cache_embeddings", "custom_embedding_file"
        }
        custom_params = {k: v for k, v in config_dict.items() if k not in known_fields}
        known_params = {k: v for k, v in config_dict.items() if k in known_fields}
        return cls(**known_params, custom_params=custom_params)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "LanguageEmbeddingsConfig":
        """Load config from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    def to_yaml(self, yaml_path: str):
        """Save config to YAML file."""
        Path(yaml_path).parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    @classmethod
    def create_simple(cls) -> "LanguageEmbeddingsConfig":
        """Create simple config (URIEL, source only)."""
        return cls(
            use_source_embedding=True,
            use_target_embedding=False,
            embedding_source="uriel",
            embedding_dim=32,
            feature_types=["syntax", "phonology"],
            integration_method="concatenate",
        )

    @classmethod
    def create_bilateral(cls) -> "LanguageEmbeddingsConfig":
        """Create bilateral config (both source and target)."""
        return cls(
            use_both=True,
            embedding_source="uriel",
            embedding_dim=64,
            feature_types=["syntax", "phonology", "morphology"],
            integration_method="concatenate",
        )


# Example configurations
EXAMPLE_CONFIGS = {
    "minimal": {
        "use_source_embedding": True,
        "embedding_source": "uriel",
        "embedding_dim": 32,
        "feature_types": ["syntax"],
        "integration_method": "concatenate",
    },
    "standard": {
        "use_source_embedding": True,
        "embedding_source": "uriel",
        "embedding_dim": 64,
        "feature_types": ["syntax", "phonology", "morphology"],
        "integration_method": "concatenate",
    },
    "bilateral": {
        "use_both": True,
        "embedding_source": "uriel",
        "embedding_dim": 64,
        "feature_types": ["syntax", "phonology", "morphology"],
        "integration_method": "concatenate",
    },
    "wals": {
        "use_source_embedding": True,
        "embedding_source": "wals",
        "embedding_dim": 64,
        "feature_types": ["syntax", "phonology"],
        "integration_method": "concatenate",
    },
}


def get_example_config(name: str) -> LanguageEmbeddingsConfig:
    """
    Get example configuration by name.

    Args:
        name: Config name ("minimal", "standard", "bilateral", "wals")

    Returns:
        LanguageEmbeddingsConfig instance
    """
    if name not in EXAMPLE_CONFIGS:
        raise ValueError(
            f"Unknown config: {name}. "
            f"Available: {list(EXAMPLE_CONFIGS.keys())}"
        )

    return LanguageEmbeddingsConfig.from_dict(EXAMPLE_CONFIGS[name])
