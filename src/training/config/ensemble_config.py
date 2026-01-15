"""Ensemble configuration."""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
from pathlib import Path
import yaml


@dataclass
class EnsembleConfig:
    """
    Configuration for ensemble translation methods.

    Supports COMET, MBR, voting, and custom ensemble strategies.
    Fully extensible for new ensemble methods.
    """

    # Ensemble method
    method: str = "comet"  # comet, mbr, voting, reranking, bilingual, backtranslation, custom

    # Models in ensemble
    model_paths: List[str] = field(default_factory=list)
    model_weights: Optional[List[float]] = None  # For weighted methods

    # COMET-based selection
    comet_model: str = "Unbabel/wmt22-comet-da"  # Default COMET model
    comet_batch_size: int = 8
    comet_use_reference: bool = False  # Reference-free by default

    # MBR (Minimum Bayes Risk) parameters
    mbr_num_samples: int = 20  # Candidates per model
    mbr_metric: str = "chrf"  # chrf, bleu, comet
    mbr_utility_function: str = "expected"  # expected, max, median

    # Voting parameters
    voting_method: str = "majority"  # majority, weighted, ranked
    voting_min_agreement: int = 2  # Minimum models that must agree

    # Reranking parameters
    reranker_features: List[str] = field(default_factory=lambda: [
        "length_ratio", "confidence", "fluency", "adequacy"
    ])
    reranker_model_path: Optional[str] = None  # Trained reranker model

    # Bilingual embedding scoring
    bilingual_embedding_model: str = "sentence-transformers/LaBSE"
    bilingual_similarity_threshold: float = 0.7

    # Back-translation validation
    backtranslation_model_path: Optional[str] = None
    backtranslation_threshold: float = 0.8  # Similarity threshold

    # Quality estimation
    quality_estimator_model: Optional[str] = None
    quality_threshold: float = 0.7  # Minimum quality score

    # Output configuration
    return_confidence_scores: bool = True
    return_all_candidates: bool = False  # Return all candidates or just best

    # Batch processing
    batch_size: int = 8
    use_gpu: bool = True
    device: str = "cuda"

    # Caching
    cache_model_outputs: bool = True
    cache_dir: Optional[str] = "cache/ensemble"

    # Fallback strategy
    fallback_method: str = "first_model"  # first_model, longest, shortest
    fallback_enabled: bool = True

    # Custom parameters
    custom_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration."""
        valid_methods = [
            "comet", "mbr", "voting", "reranking",
            "bilingual", "backtranslation", "custom"
        ]
        if self.method not in valid_methods:
            raise ValueError(f"Invalid method: {self.method}. Choose from {valid_methods}")

        if len(self.model_paths) < 2:
            raise ValueError("At least 2 models required for ensemble")

        # Validate weights if provided
        if self.model_weights is not None:
            if len(self.model_weights) != len(self.model_paths):
                raise ValueError("Number of weights must match number of models")
            if abs(sum(self.model_weights) - 1.0) > 0.01:
                raise ValueError("Weights must sum to 1.0")

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "EnsembleConfig":
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "EnsembleConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)

    def to_yaml(self, yaml_path: str):
        """Save configuration to YAML file."""
        Path(yaml_path).parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, 'w') as f:
            yaml.dump(asdict(self), f, default_flow_style=False)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
    def create_preset(cls, method: str, model_paths: List[str]) -> "EnsembleConfig":
        """
        Create ensemble configuration from preset.

        Presets:
        - comet: COMET-based quality estimation (best quality)
        - mbr: Minimum Bayes Risk consensus
        - voting: Simple majority voting (fastest)
        - quality: Combined quality metrics
        """
        presets = {
            "comet": {
                "method": "comet",
                "comet_use_reference": False,
                "comet_batch_size": 8,
            },
            "mbr": {
                "method": "mbr",
                "mbr_num_samples": 20,
                "mbr_metric": "chrf",
            },
            "voting": {
                "method": "voting",
                "voting_method": "majority",
                "voting_min_agreement": 2,
            },
            "quality": {
                "method": "reranking",
                "reranker_features": [
                    "length_ratio", "confidence", "fluency",
                    "adequacy", "comet_score"
                ],
            },
        }

        if method not in presets:
            raise ValueError(f"Unknown preset: {method}. Choose from {list(presets.keys())}")

        config = presets[method].copy()
        config["model_paths"] = model_paths

        return cls(**config)
