"""Configuration for SMT-inspired quality components."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import yaml
from pathlib import Path


@dataclass
class QualityConfig:
    """
    Configuration for quality estimation components.

    Attributes:
        use_quality_estimation: Whether to use quality estimation
        use_adequacy: Whether to use adequacy component
        use_fluency: Whether to use fluency component
        adequacy_method: Method for adequacy estimation
        fluency_method: Method for fluency estimation
        adequacy_model: Model for adequacy (if applicable)
        fluency_model: Model for fluency (if applicable)
        use_quality_weighted_loss: Use quality-weighted loss for training
        adequacy_weight: Weight for adequacy loss
        fluency_weight: Weight for fluency loss
        quality_threshold: Minimum quality threshold for filtering
        normalize_scores: Whether to normalize quality scores
        cache_quality_scores: Whether to cache computed scores
    """

    # Enable/disable quality estimation
    use_quality_estimation: bool = False
    use_adequacy: bool = True
    use_fluency: bool = True

    # Adequacy configuration
    adequacy_method: str = "sentence_embedding"
    # Options:
    # - sentence_embedding: Use sentence embeddings similarity
    # - entailment: Use entailment model
    # - word_alignment: Use word alignment scores

    adequacy_model: str = "sentence-transformers/LaBSE"
    # Options for sentence_embedding:
    # - sentence-transformers/LaBSE (multilingual)
    # - sentence-transformers/paraphrase-multilingual-mpnet-base-v2
    # Options for entailment:
    # - microsoft/mdeberta-v3-base-xnli (multilingual NLI)
    # - joeddav/xlm-roberta-large-xnli

    # Fluency configuration
    fluency_method: str = "perplexity"
    # Options:
    # - perplexity: Use language model perplexity
    # - parse_based: Use parse tree metrics
    # - grammar_checker: Use grammar checker

    fluency_model: str = "gpt2"
    # Options for perplexity:
    # - gpt2 (small, fast)
    # - facebook/mbart-large-50
    # - bigscience/bloom-560m
    # Options for parse_based:
    # - stanza (uses dependency parser)

    # Quality-weighted training
    use_quality_weighted_loss: bool = False
    adequacy_weight: float = 0.5
    fluency_weight: float = 0.5

    # Quality filtering
    quality_threshold: float = 0.0  # No filtering by default
    normalize_scores: bool = True

    # Caching
    cache_quality_scores: bool = True
    quality_cache_dir: str = "cache/quality"

    # Advanced options
    batch_size: int = 32
    device: str = "cuda"

    # Custom parameters
    custom_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration."""
        if not self.use_adequacy and not self.use_fluency:
            raise ValueError(
                "Must enable at least one of: use_adequacy, use_fluency"
            )

        valid_adequacy_methods = [
            "sentence_embedding",
            "entailment",
            "word_alignment"
        ]
        if self.adequacy_method not in valid_adequacy_methods:
            raise ValueError(
                f"Invalid adequacy_method: {self.adequacy_method}. "
                f"Must be one of: {valid_adequacy_methods}"
            )

        valid_fluency_methods = [
            "perplexity",
            "parse_based",
            "grammar_checker"
        ]
        if self.fluency_method not in valid_fluency_methods:
            raise ValueError(
                f"Invalid fluency_method: {self.fluency_method}. "
                f"Must be one of: {valid_fluency_methods}"
            )

        if self.adequacy_weight + self.fluency_weight != 1.0:
            raise ValueError(
                f"Weights must sum to 1.0, got {self.adequacy_weight + self.fluency_weight}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "use_quality_estimation": self.use_quality_estimation,
            "use_adequacy": self.use_adequacy,
            "use_fluency": self.use_fluency,
            "adequacy_method": self.adequacy_method,
            "fluency_method": self.fluency_method,
            "adequacy_model": self.adequacy_model,
            "fluency_model": self.fluency_model,
            "use_quality_weighted_loss": self.use_quality_weighted_loss,
            "adequacy_weight": self.adequacy_weight,
            "fluency_weight": self.fluency_weight,
            "quality_threshold": self.quality_threshold,
            "normalize_scores": self.normalize_scores,
            "cache_quality_scores": self.cache_quality_scores,
            "quality_cache_dir": self.quality_cache_dir,
            "batch_size": self.batch_size,
            "device": self.device,
            **self.custom_params,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "QualityConfig":
        """Create config from dictionary."""
        known_fields = {
            "use_quality_estimation", "use_adequacy", "use_fluency",
            "adequacy_method", "fluency_method", "adequacy_model", "fluency_model",
            "use_quality_weighted_loss", "adequacy_weight", "fluency_weight",
            "quality_threshold", "normalize_scores", "cache_quality_scores",
            "quality_cache_dir", "batch_size", "device"
        }
        custom_params = {k: v for k, v in config_dict.items() if k not in known_fields}
        known_params = {k: v for k, v in config_dict.items() if k in known_fields}
        return cls(**known_params, custom_params=custom_params)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "QualityConfig":
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
    def create_simple(cls) -> "QualityConfig":
        """Create simple config (adequacy only)."""
        return cls(
            use_quality_estimation=True,
            use_adequacy=True,
            use_fluency=False,
            adequacy_method="sentence_embedding",
            adequacy_model="sentence-transformers/LaBSE",
            adequacy_weight=1.0,
            fluency_weight=0.0,
        )

    @classmethod
    def create_full(cls) -> "QualityConfig":
        """Create full config (adequacy + fluency)."""
        return cls(
            use_quality_estimation=True,
            use_adequacy=True,
            use_fluency=True,
            adequacy_method="sentence_embedding",
            fluency_method="perplexity",
            adequacy_model="sentence-transformers/LaBSE",
            fluency_model="gpt2",
            use_quality_weighted_loss=True,
            adequacy_weight=0.5,
            fluency_weight=0.5,
        )


# Example configurations
EXAMPLE_CONFIGS = {
    "adequacy_only": {
        "use_quality_estimation": True,
        "use_adequacy": True,
        "use_fluency": False,
        "adequacy_method": "sentence_embedding",
        "adequacy_model": "sentence-transformers/LaBSE",
        "adequacy_weight": 1.0,
        "fluency_weight": 0.0,
    },
    "fluency_only": {
        "use_quality_estimation": True,
        "use_adequacy": False,
        "use_fluency": True,
        "fluency_method": "perplexity",
        "fluency_model": "gpt2",
        "adequacy_weight": 0.0,
        "fluency_weight": 1.0,
    },
    "balanced": {
        "use_quality_estimation": True,
        "use_adequacy": True,
        "use_fluency": True,
        "adequacy_method": "sentence_embedding",
        "fluency_method": "perplexity",
        "adequacy_model": "sentence-transformers/LaBSE",
        "fluency_model": "gpt2",
        "use_quality_weighted_loss": True,
        "adequacy_weight": 0.5,
        "fluency_weight": 0.5,
    },
}


def get_example_config(name: str) -> QualityConfig:
    """
    Get example configuration by name.

    Args:
        name: Config name ("adequacy_only", "fluency_only", "balanced")

    Returns:
        QualityConfig instance
    """
    if name not in EXAMPLE_CONFIGS:
        raise ValueError(
            f"Unknown config: {name}. "
            f"Available: {list(EXAMPLE_CONFIGS.keys())}"
        )

    return QualityConfig.from_dict(EXAMPLE_CONFIGS[name])
