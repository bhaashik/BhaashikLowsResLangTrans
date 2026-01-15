"""Model merging configuration."""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
from pathlib import Path
import yaml


@dataclass
class MergingConfig:
    """
    Configuration for model merging strategies.

    Supports TIES, DARE, Model Soups, LoRA merging, and custom methods.
    Fully extensible for new merging algorithms.
    """

    # Merging method
    method: str = "ties"  # ties, dare, model_soups, lora_merging, task_arithmetic, fisher, custom

    # Models to merge
    model_paths: List[str] = field(default_factory=list)
    base_model_path: Optional[str] = None  # Required for delta-based methods

    # Output configuration
    output_dir: str = "models/merged"
    save_format: str = "full"  # full, adapter_only

    # TIES-Merging parameters
    ties_density: float = 0.8  # Keep top 80% of parameters
    ties_majority_sign_method: str = "total"  # total, frequency

    # DARE parameters
    dare_drop_rate: float = 0.9  # Drop 90% of delta parameters
    dare_rescale_mode: str = "adaptive"  # adaptive, constant, layer_wise
    dare_rescale_factor: float = 1.0  # For constant mode

    # Model Soups parameters
    soup_method: str = "greedy"  # uniform, greedy
    soup_eval_metric: str = "bleu"  # Metric for greedy soup
    soup_max_models: Optional[int] = None  # Limit number of models in soup

    # Task Arithmetic parameters
    task_arithmetic_scaling: float = 1.0  # Scaling factor for task vectors
    task_arithmetic_operation: str = "add"  # add, subtract, negate

    # Fisher Merging parameters
    fisher_weighting_method: str = "fisher"  # fisher, regret
    fisher_normalize_weights: bool = True

    # LoRA Merging parameters
    lora_merge_method: str = "linear"  # linear, learned, mixture
    lora_merge_weights: Optional[List[float]] = None  # Custom weights for linear
    lora_scaling: float = 1.0

    # Git Re-Basin parameters
    rebasin_enabled: bool = False
    rebasin_max_iterations: int = 100
    rebasin_tolerance: float = 1e-4

    # Knowledge Distillation parameters
    distillation_temperature: float = 2.0
    distillation_alpha: float = 0.5  # Weight for distillation loss

    # Analysis options
    analyze_conflicts: bool = False
    conflict_analysis_output: Optional[str] = None

    # Device configuration
    device: str = "cuda"
    torch_dtype: str = "float32"  # float32, float16, bfloat16

    # Custom parameters
    custom_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration."""
        valid_methods = [
            "ties", "dare", "model_soups", "lora_merging",
            "task_arithmetic", "fisher", "git_rebasin",
            "distillation", "custom"
        ]
        if self.method not in valid_methods:
            raise ValueError(f"Invalid method: {self.method}. Choose from {valid_methods}")

        # Validate parameters based on method
        if self.method in ["ties", "dare", "task_arithmetic", "fisher"]:
            if self.base_model_path is None:
                raise ValueError(f"{self.method} requires base_model_path")

        if len(self.model_paths) < 2:
            raise ValueError("At least 2 models required for merging")

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "MergingConfig":
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "MergingConfig":
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
    def create_preset(cls, method: str, model_paths: List[str], base_model: Optional[str] = None) -> "MergingConfig":
        """
        Create merging configuration from preset.

        Presets:
        - ties: TIES-Merging with optimal density
        - dare: DARE with high drop rate
        - model_soups: Greedy soup selection
        - ties_dare: Combined TIES + DARE (best quality)
        """
        presets = {
            "ties": {
                "method": "ties",
                "ties_density": 0.8,
            },
            "dare": {
                "method": "dare",
                "dare_drop_rate": 0.9,
                "dare_rescale_mode": "adaptive",
            },
            "model_soups": {
                "method": "model_soups",
                "soup_method": "greedy",
                "soup_eval_metric": "bleu",
            },
            "ties_dare": {
                "method": "ties",
                "ties_density": 0.9,
                "custom_params": {"apply_dare_after_ties": True, "dare_drop_rate": 0.95},
            },
        }

        if method not in presets:
            raise ValueError(f"Unknown preset: {method}. Choose from {list(presets.keys())}")

        config = presets[method].copy()
        config["model_paths"] = model_paths
        config["base_model_path"] = base_model

        return cls(**config)
