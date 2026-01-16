"""
Configuration for linguistic features.

Defines all parameters for using linguistic features in MT training.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import yaml
from pathlib import Path


@dataclass
class LinguisticFeaturesConfig:
    """
    Configuration for linguistic features in MT.

    Attributes:
        use_source_parse: Whether to use source-side parse
        use_target_parse: Whether to use target-side parse
        use_both: Whether to use both source and target parses
        parser: Parser name ("stanza", "spacy", "trankit", "udpipe")
        parser_batch_size: Batch size for parsing
        features: List of feature types to extract
        integration_method: How to integrate features into model
        encoding_dim: Dimension of encoded features
        use_graph_encoder: Whether to use GNN for parse encoding
        cache_parses: Whether to cache parsed trees
        parse_cache_dir: Directory for parse cache
        custom_params: Additional custom parameters
    """

    # Enable/disable features
    use_source_parse: bool = False
    use_target_parse: bool = False
    use_both: bool = False

    # Parser selection
    parser: str = "stanza"  # stanza, spacy, trankit, udpipe
    parser_batch_size: int = 32
    parser_use_gpu: bool = True

    # Feature types
    features: List[str] = field(default_factory=lambda: [
        "dependency_labels",  # DEPREL tags
        "pos_tags",           # UPOS tags
        "tree_depth",         # Depth in parse tree
        "head_distance",      # Distance to head
    ])

    # Integration method
    integration_method: str = "encoder_augmentation"
    # Options:
    # - encoder_augmentation: Concatenate with word embeddings
    # - attention_injection: Bias attention with parse structure
    # - decoder_constraint: Constrain decoder with target parse

    # Feature encoding
    encoding_dim: int = 128
    use_graph_encoder: bool = False  # Use GNN for parse encoding

    # Caching (speeds up training with fixed data)
    cache_parses: bool = True
    parse_cache_dir: str = "cache/parses"

    # Custom parameters
    custom_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration."""
        if self.use_both:
            self.use_source_parse = True
            self.use_target_parse = True

        if not self.use_source_parse and not self.use_target_parse:
            raise ValueError(
                "Must enable at least one of: use_source_parse, use_target_parse"
            )

        valid_parsers = ["stanza", "spacy", "trankit", "udpipe"]
        if self.parser not in valid_parsers:
            raise ValueError(
                f"Invalid parser: {self.parser}. "
                f"Must be one of: {valid_parsers}"
            )

        valid_methods = [
            "encoder_augmentation",
            "attention_injection",
            "decoder_constraint"
        ]
        if self.integration_method not in valid_methods:
            raise ValueError(
                f"Invalid integration_method: {self.integration_method}. "
                f"Must be one of: {valid_methods}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "use_source_parse": self.use_source_parse,
            "use_target_parse": self.use_target_parse,
            "use_both": self.use_both,
            "parser": self.parser,
            "parser_batch_size": self.parser_batch_size,
            "parser_use_gpu": self.parser_use_gpu,
            "features": self.features,
            "integration_method": self.integration_method,
            "encoding_dim": self.encoding_dim,
            "use_graph_encoder": self.use_graph_encoder,
            "cache_parses": self.cache_parses,
            "parse_cache_dir": self.parse_cache_dir,
            **self.custom_params,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "LinguisticFeaturesConfig":
        """Create config from dictionary."""
        known_fields = {
            "use_source_parse", "use_target_parse", "use_both",
            "parser", "parser_batch_size", "parser_use_gpu",
            "features", "integration_method", "encoding_dim",
            "use_graph_encoder", "cache_parses", "parse_cache_dir"
        }
        custom_params = {k: v for k, v in config_dict.items() if k not in known_fields}
        known_params = {k: v for k, v in config_dict.items() if k in known_fields}
        return cls(**known_params, custom_params=custom_params)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "LinguisticFeaturesConfig":
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
    def create_simple(cls) -> "LinguisticFeaturesConfig":
        """Create simple config (source parse only, basic features)."""
        return cls(
            use_source_parse=True,
            use_target_parse=False,
            parser="stanza",
            features=["dependency_labels", "pos_tags"],
            integration_method="encoder_augmentation",
            encoding_dim=64,
            use_graph_encoder=False,
        )

    @classmethod
    def create_advanced(cls) -> "LinguisticFeaturesConfig":
        """Create advanced config (both sides, GNN, all features)."""
        return cls(
            use_both=True,
            parser="stanza",
            features=[
                "dependency_labels",
                "pos_tags",
                "tree_depth",
                "head_distance",
                "is_root",
                "num_children",
            ],
            integration_method="attention_injection",
            encoding_dim=256,
            use_graph_encoder=True,
        )


# Example configurations

EXAMPLE_CONFIGS = {
    "minimal": {
        "use_source_parse": True,
        "parser": "stanza",
        "features": ["dependency_labels"],
        "integration_method": "encoder_augmentation",
        "encoding_dim": 64,
    },
    "standard": {
        "use_source_parse": True,
        "parser": "stanza",
        "features": ["dependency_labels", "pos_tags", "tree_depth"],
        "integration_method": "encoder_augmentation",
        "encoding_dim": 128,
    },
    "full": {
        "use_both": True,
        "parser": "stanza",
        "features": [
            "dependency_labels",
            "pos_tags",
            "tree_depth",
            "head_distance",
        ],
        "integration_method": "attention_injection",
        "encoding_dim": 256,
        "use_graph_encoder": True,
    },
}


def get_example_config(name: str) -> LinguisticFeaturesConfig:
    """
    Get example configuration by name.

    Args:
        name: Config name ("minimal", "standard", "full")

    Returns:
        LinguisticFeaturesConfig instance
    """
    if name not in EXAMPLE_CONFIGS:
        raise ValueError(
            f"Unknown config: {name}. "
            f"Available: {list(EXAMPLE_CONFIGS.keys())}"
        )

    return LinguisticFeaturesConfig.from_dict(EXAMPLE_CONFIGS[name])
