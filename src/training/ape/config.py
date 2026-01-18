"""Configuration for Automatic Post-Editing."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import yaml
from pathlib import Path


@dataclass
class APEConfig:
    """
    Configuration for Automatic Post-Editing.

    Attributes:
        use_ape: Whether to use APE
        ape_method: APE method to use
        quality_threshold: Only refine translations below this quality
        max_iterations: Maximum refinement iterations
        min_quality_improvement: Minimum quality gain to continue
        use_quality_estimation: Use quality estimator to decide when to stop
        beam_size: Beam size for generation
        temperature: Sampling temperature for refinement
        top_p: Nucleus sampling parameter
        batch_size: Batch size for APE
        device: Device to run APE on
    """

    # Enable/disable APE
    use_ape: bool = False

    # APE method
    ape_method: str = "iterative"
    # Options:
    # - iterative: Iterative refinement with context
    # - backtranslation: Round-trip verification and refinement
    # - multipass: Multiple passes with different strategies
    # - ensemble: Ensemble of different APE methods

    # Quality-based triggering
    quality_threshold: float = 0.7
    # Only apply APE to translations with quality < threshold

    # Iterative refinement parameters
    max_iterations: int = 3
    min_quality_improvement: float = 0.01
    # Stop if improvement < this threshold

    # Quality estimation
    use_quality_estimation: bool = True
    # Use quality estimator to guide APE

    # Generation parameters
    beam_size: int = 5
    temperature: float = 1.0
    top_p: float = 0.9
    length_penalty: float = 1.0

    # Processing
    batch_size: int = 16
    device: str = "cuda"

    # Back-translation specific
    backtranslation_model: Optional[str] = None
    # Model for back-translation (target→source)

    # Multi-pass specific
    num_passes: int = 2
    pass_strategies: List[str] = field(default_factory=lambda: ["beam", "sampling"])
    # Strategies for each pass

    # Custom parameters
    custom_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration."""
        valid_methods = ["iterative", "backtranslation", "multipass", "ensemble"]
        if self.ape_method not in valid_methods:
            raise ValueError(
                f"Invalid ape_method: {self.ape_method}. "
                f"Must be one of: {valid_methods}"
            )

        if self.quality_threshold < 0.0 or self.quality_threshold > 1.0:
            raise ValueError(
                f"quality_threshold must be in [0, 1], got {self.quality_threshold}"
            )

        if self.max_iterations < 1:
            raise ValueError(
                f"max_iterations must be >= 1, got {self.max_iterations}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "use_ape": self.use_ape,
            "ape_method": self.ape_method,
            "quality_threshold": self.quality_threshold,
            "max_iterations": self.max_iterations,
            "min_quality_improvement": self.min_quality_improvement,
            "use_quality_estimation": self.use_quality_estimation,
            "beam_size": self.beam_size,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "length_penalty": self.length_penalty,
            "batch_size": self.batch_size,
            "device": self.device,
            "backtranslation_model": self.backtranslation_model,
            "num_passes": self.num_passes,
            "pass_strategies": self.pass_strategies,
            **self.custom_params,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "APEConfig":
        """Create config from dictionary."""
        known_fields = {
            "use_ape", "ape_method", "quality_threshold", "max_iterations",
            "min_quality_improvement", "use_quality_estimation",
            "beam_size", "temperature", "top_p", "length_penalty",
            "batch_size", "device", "backtranslation_model",
            "num_passes", "pass_strategies"
        }
        custom_params = {k: v for k, v in config_dict.items() if k not in known_fields}
        known_params = {k: v for k, v in config_dict.items() if k in known_fields}
        return cls(**known_params, custom_params=custom_params)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "APEConfig":
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
    def create_simple(cls) -> "APEConfig":
        """Create simple APE config (iterative refinement)."""
        return cls(
            use_ape=True,
            ape_method="iterative",
            quality_threshold=0.7,
            max_iterations=2,
            use_quality_estimation=True,
        )

    @classmethod
    def create_aggressive(cls) -> "APEConfig":
        """Create aggressive APE config (more iterations)."""
        return cls(
            use_ape=True,
            ape_method="multipass",
            quality_threshold=0.8,
            max_iterations=5,
            num_passes=3,
            use_quality_estimation=True,
        )


# Example configurations
EXAMPLE_CONFIGS = {
    "simple": {
        "use_ape": True,
        "ape_method": "iterative",
        "quality_threshold": 0.7,
        "max_iterations": 2,
        "use_quality_estimation": True,
    },
    "backtranslation": {
        "use_ape": True,
        "ape_method": "backtranslation",
        "quality_threshold": 0.6,
        "max_iterations": 1,
        "use_quality_estimation": True,
    },
    "multipass": {
        "use_ape": True,
        "ape_method": "multipass",
        "quality_threshold": 0.7,
        "num_passes": 2,
        "pass_strategies": ["beam", "sampling"],
        "use_quality_estimation": True,
    },
}


def get_example_config(name: str) -> APEConfig:
    """
    Get example configuration by name.

    Args:
        name: Config name ("simple", "backtranslation", "multipass")

    Returns:
        APEConfig instance
    """
    if name not in EXAMPLE_CONFIGS:
        raise ValueError(
            f"Unknown config: {name}. "
            f"Available: {list(EXAMPLE_CONFIGS.keys())}"
        )

    return APEConfig.from_dict(EXAMPLE_CONFIGS[name])
